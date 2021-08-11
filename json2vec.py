from __future__ import annotations
from typing import Any, Callable, Union, Optional, Tuple, Sequence, List, TypeVar, Mapping, Dict, cast, MutableSequence
from abc import ABC, abstractmethod
import ujson as json
import numpy as np

import torch
import torch.nn as nn

import numbers
import string
from collections import defaultdict

ALL_CHARACTERS = string.printable
N_CHARACTERS = len(ALL_CHARACTERS)


JSONType = Union[str, numbers.Number, Dict[str, Any], List]
TensorPair = Tuple[torch.Tensor, torch.Tensor]
PathSeq = List[Union[str, numbers.Number]]


class KeyDefaultDict(defaultdict):
    default_factory: Optional[Callable[[str], Any]]

    def __missing__(self, key: str) -> Any:
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


# module for childsumtreelstm
class JSONNN(nn.Module, ABC):
    def __init__(self, mem_dim: int = 128, decode_json: bool = False):
        super(JSONNN, self).__init__()
        self.mem_dim = mem_dim
        self.device = None
        self.decode_json = decode_json

    def add_module(self, name: str, module):
        module.to(self.device)
        super().add_module(name, module)

    def forward(self, node_as_json_str: str):
        if not self.decode_json:
            node_as_json_value = json.loads(node_as_json_str)
        else:
            node_as_json_value = node_as_json_str

        return self.embed_node(node_as_json_value, path=["___root___"])
        # return self.embedNode(node_as_json_str, path=["___root___"])[0]


    def embed_node(self, node_as_json_value: JSONType, path: PathSeq=None) -> Optional[TensorPair]:
        path = path or []
        if isinstance(node_as_json_value, dict):  # DICT
            child_states = [child_state for child_name, child in node_as_json_value.items()
                            if (child_state := self.embed_node(child, path + [child_name])) is not None]
            if not child_states:
                return None
            return self.embed_object(child_states, path)

        if isinstance(node_as_json_value, list):  # DICT
            child_states = [child_state for i, child in enumerate(node_as_json_value)
                             if (child_state := self.embed_node(child, path + [i]))]

            if not child_states:
                return None
            return self.embed_array(child_states, path)


        elif isinstance(node_as_json_value, str):  # STRING
            return self.embed_string(node_as_json_value, path)

        elif isinstance(node_as_json_value, numbers.Number):  # NUMBER
            return self.embed_number(node_as_json_value, path)

        else:
            # not impl error
            return None

    @abstractmethod
    def embed_array(self, child_states: Sequence[TensorPair], path: PathSeq) -> Optional[TensorPair]:
        """Embeds JSON arrays"""

    @abstractmethod
    def embed_object(self, child_states: Sequence[TensorPair], path: PathSeq) -> Optional[TensorPair]:
        """Embeds JSON objects"""

    @abstractmethod
    def embed_string(self, node: str, path: PathSeq) -> Optional[TensorPair]:
        """Embeds JSON strings"""

    @abstractmethod
    def embed_number(self, node: numbers.Number, path: PathSeq) -> Optional[TensorPair]:
        """Embeds JSON numbers"""

    def _canonical(self, path: PathSeq) -> Tuple[str, ...]:
        # Restrict to last two elements and replace ints with placeholder '___list___'
        return tuple(
                p if not isinstance(p, numbers.Number) else '___list___'
                for p in path[-2:]
        )


class JSONTreeLSTM(JSONNN):
    def __init__(self, mem_dim=128, decode_json=False,
                 tie_weights_containers=False,
                 tie_weights_primitives=False,
                 homogenous_types=False):
        super(JSONTreeLSTM, self).__init__(mem_dim, decode_json)

        self.iouh = KeyDefaultDict(self._newiouh)
        self.fh = KeyDefaultDict(self._newfh)
        self.lout = KeyDefaultDict(self._newlout)

        self.lstms = KeyDefaultDict(self._new_lstm)

        self.string_encoder = nn.Embedding(N_CHARACTERS, self.mem_dim).to(self.device)
        self.string_rnn = KeyDefaultDict(self._new_string_rnn)
        # nn.LSTM(self.mem_dim, self.mem_dim, 1)

        self.number_embeddings = KeyDefaultDict(self._new_number)
        self.number_stats = defaultdict(lambda: [])

        self.tie_primitives = tie_weights_primitives
        self.tie_containers = tie_weights_containers
        self.homogenous_types = homogenous_types

    def forward(self, node_as_json_str: str) -> torch.Tensor:
        result = super().forward(node_as_json_str)
        if result is None: return torch.cat([self._init_c()] * 2, 1)
        return torch.cat(result, 1)

    def _init_c(self):
        return torch.zeros(1, self.mem_dim, requires_grad=True, device=self.device)

    def _newiouh(self, key):
        layer = nn.Linear(self.mem_dim, self.mem_dim * 3)
        self.add_module(str(key) + "_iouh", layer)
        return layer

    def _newfh(self, key):
        layer = nn.Linear(self.mem_dim, self.mem_dim)
        self.add_module(str(key) + "_fh", layer)
        return layer

    def _newlout(self, key):
        layer = nn.Linear(self.mem_dim, self.mem_dim)
        self.add_module(str(key) + "_lout", layer)
        return layer

    def embed_object(self, child_states: Sequence[TensorPair], path: PathSeq) -> Optional[TensorPair]:
        canon_path = self._canonical(path) if not self.tie_containers else "___default___"

        hs = []
        fcs = []
        for child_c, child_h in child_states:
            hs.append(child_h)

            f = self.fh[self._canonical(path)](child_h)
            # f = self.fh["___default___"](child_h)
            fc = torch.mul(torch.sigmoid(f), child_c)
            fcs.append(fc)

        if not hs:
            return None

        hs_bar = torch.sum(torch.cat(hs), dim=0, keepdim=True)
        iou = self.iouh[self._canonical(path)](hs_bar)
        # iou = self.iouh["___default___"](hs_bar)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        c = torch.mul(i, u) + torch.sum(torch.cat(fcs), dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))

        h_hat = self.lout[canon_path](h)

        return c, h_hat

    def _new_lstm(self, key: str) -> nn.Module:
        lstm = nn.LSTMCell(input_size=self.mem_dim * 2, hidden_size=self.mem_dim)
        self.add_module(str(key) + "_ArrayLSTM", lstm)
        return lstm

    def embed_array(self, child_states: Sequence[TensorPair], path: PathSeq) -> Optional[TensorPair]:
        if self.homogenous_types:
            return self.embed_object(child_states, path)

        canon_path = self._canonical(path) if not self.tie_containers else "___default___"
        lstm = self.lstms[canon_path]
        c = self._init_c()
        h = self._init_c()
        for child_c, child_h in child_states:
            child_ch = torch.cat([child_c, child_h], dim=1)
            c, h = lstm(child_ch, (c, h))
        return c, h

    def _new_string_rnn(self, key: str) -> nn.Module:
        lstm = nn.LSTM(self.mem_dim, self.mem_dim, 1)
        self.add_module(str(key) + "_StringLSTM", lstm)
        return lstm

    def embed_string(self, string: str, path: PathSeq) -> TensorPair:
        canon_path = self._canonical(path) if not self.tie_primitives else "___default___"
        if string == "":
            return self._init_c(), self._init_c()

        tensor_input = torch.tensor(
                [ALL_CHARACTERS.index(char) for char in string if char in ALL_CHARACTERS],
                device=self.device,
        ).long()
	
        encoded_input = self.string_encoder(tensor_input.view(1, -1)).view(-1, 1, self.mem_dim)
        output, hidden = self.string_rnn[canon_path](encoded_input)
        return self._init_c(), hidden[0].mean(dim=1)
        # hidden[1].mean(dim=1), hidden[0].mean(dim=1)

    def _new_number(self, key: str) -> nn.Module:
        layer = nn.Linear(1, self.mem_dim)
        self.add_module(str(key) + "_number", layer)
        return layer

    def embed_number(self, num: numbers.Number, path: PathSeq) -> TensorPair:
        if self.homogenous_types:
            return self.embed_string(str(num), path)

        canon_path = self._canonical(path) if not self.tie_primitives else "___default___"

        # TODO: Implement actual moving average
        """
        if len(self.number_stats[canon_path]) < 100:
            self.number_stats[canon_path].append(num)

        num -= np.mean(self.number_stats[canon_path])

        if len(self.number_stats[canon_path]) > 3:
            std = np.std(self.number_stats[canon_path])
            if not np.allclose(std, 0.0):
                num /= std
        """

        encoded = self.number_embeddings[canon_path](torch.tensor([[num]], dtype=torch.float32, device=self.device))
        return self._init_c(), encoded


class SetJSONNN(JSONNN):
    def __init__(self, mem_dim=128,
                 tie_weights_containers=False,
                 tie_weights_primitives=False,
                 homogenous_types=False):
        super(SetJSONNN, self).__init__(mem_dim)

        self.mem_dim = mem_dim
        self.embedder = KeyDefaultDict(self._new_embedder)
        self.out = KeyDefaultDict(self._new_out)

        self.string_encoder = nn.Embedding(N_CHARACTERS, self.mem_dim)
        self.string_rnn = KeyDefaultDict(self._new_string_rnn)

        self.number_embeddings = KeyDefaultDict(self._new_number)
        self.number_stats = defaultdict(lambda: [])

        self.tie_primitives = tie_weights_primitives
        self.tie_containers = tie_weights_containers
        self.homogenous_types = homogenous_types

    def forward(self, node_as_json_str):
        if isinstance(node_as_json_str, str):
            try:
                node_as_json_str = json.loads(node_as_json_str)
            except:
                raise Exception(node_as_json_str)

        node_as_json_str = self.embed_node(node_as_json_str, path=["___root___"])
        if node_as_json_str is None: return self._init_c()
        return node_as_json_str

    def _init_c(self):
        return torch.empty(1, self.mem_dim).fill_(0.).requires_grad_()

    def _new_embedder(self, key):
        l1 = torch.nn.Linear(self.mem_dim, self.mem_dim)
        l2 = torch.nn.Linear(self.mem_dim, self.mem_dim)
        self.add_module(str(key) + "_l1", l1)
        self.add_module(str(key) + "_l2", l2)
        model = torch.nn.Sequential(l1, torch.nn.ReLU(), l2)
        return model

    def _new_out(self, key):
        l1 = torch.nn.Linear(self.mem_dim, self.mem_dim)
        self.add_module(str(key) + "_out_l1", l1)
        # l2 = torch.nn.Linear(self.mem_dim, self.mem_dim)
        # self.add_module(str(key)+"_out_l2", l2)
        # l3 = torch.nn.Linear(self.mem_dim, self.mem_dim)
        # self.add_module(str(key)+"_out_l3", l3)
        model = torch.nn.Sequential(torch.nn.ReLU(),
                                    l1, torch.nn.ReLU(), )
        # l2, torch.nn.ReLU(),
        # l3, torch.nn.ReLU() )
        return model

    def embed_object(self, child_states, path):
        canon_path = self._canonical(path) if not self.tie_containers else "___default___"

        c_ts = []
        for c in child_states:
            c_t = self.embedder["___default___"](c)
            c_ts.append(c_t)

        if not c_ts:
            return None

        return self.out[canon_path](torch.max(torch.cat(c_ts),
                                              dim=0, keepdim=True)[0])

    def embed_array(self, child_states, path):
        canon_path = self._canonical(path) if not self.tie_containers else "___default___"

        c_ts = []
        for c in child_states:
            c_t = self.embedder[canon_path](c)
            c_ts.append(c_t)

        if not c_ts:
            return None

        return self.out[canon_path](torch.max(torch.cat(c_ts),
                                              dim=0, keepdim=True)[0])

    def _new_string_rnn(self, key):
        lstm = nn.LSTM(self.mem_dim, self.mem_dim, 1)
        self.add_module(str(key) + "_StringLSTM", lstm)
        return lstm

    def embed_string(self, string, path):
        canon_path = self._canonical(path) if not self.tie_primitives else "___default___"
        if string == "":
            return self._init_c()

        tensor_input = torch.zeros(len(string)).long()
        for c in range(len(string)):
            try:
                tensor_input[c] = ALL_CHARACTERS.index(string[c])
            except:
                continue

        encoded_input = self.string_encoder(tensor_input.view(1, -1)).view(-1, 1, self.mem_dim)
        output, hidden = self.string_rnn[canon_path](encoded_input)
        return hidden[0].mean(dim=1)

    def _new_number(self, key):
        layer = nn.Linear(1, self.mem_dim)
        self.add_module(str(key), layer)
        return layer

    def embed_number(self, num, path):
        if self.homogenous_types:
            return self.embed_string(str(num), path)

        canon_path = self._canonical(path) if not self.tie_primitives else "___default___"

        if len(self.number_stats[canon_path]) < 100:
            self.number_stats[canon_path].append(num)

        num -= np.mean(self.number_stats[canon_path])

        if len(self.number_stats[canon_path]) > 3:
            std = np.std(self.number_stats[canon_path])
            if not np.allclose(std, 0.0):
                num /= std

        encoded = self.number_embeddings[canon_path](torch.Tensor([[num]]))
        return encoded


class FlatJSONNN(JSONNN):
    def __init__(self, mem_dim=128):
        super(FlatJSONNN, self).__init__(mem_dim)

        self.string_encoder = nn.Embedding(N_CHARACTERS, self.mem_dim)
        self.string_rnn = KeyDefaultDict(self._new_string_rnn)

        self.numberEmbeddings = KeyDefaultDict(self._new_number)
        self.numberStats = defaultdict(lambda: [])

    def forward(self, node_as_json_str):
        return self.embed_node(node_as_json_str, path=["___root___"])

    def embed_object(self, child_states, path):
        return torch.sum(torch.cat(child_states), dim=0, keepdim=True)

    def embed_array(self, child_states, path):
        return torch.sum(torch.cat(child_states), dim=0, keepdim=True)

    def _new_string_rnn(self, key):
        lstm = nn.LSTM(self.mem_dim, self.mem_dim, 1)
        self.add_module(str(key) + "_StringLSTM", lstm)
        return lstm

    def embed_string(self, string, path):
        if string == "":
            return torch.empty(1, self.mem_dim).fill_(0.).requires_grad_()

        tensor_input = torch.zeros(len(string)).long()
        for c in range(len(string)):
            try:
                tensor_input[c] = ALL_CHARACTERS.index(string[c])
            except:
                continue
        encoded_input = self.string_encoder(tensor_input.view(1, -1)).view(-1, 1, self.mem_dim)
        output, hidden = self.string_rnn[self._canonical(path)](encoded_input)
        return hidden[0][:, -1, :]

    def _new_number(self, key):
        layer = nn.Linear(1, self.mem_dim)
        self.add_module(str(key), layer)
        return layer

    def embed_number(self, num, path):
        if len(self.numberStats[self._canonical(path)]) < 100:
            self.numberStats[self._canonical(path)].append(num)

        num -= np.mean(self.numberStats[self._canonical(path)])

        if len(self.numberStats[self._canonical(path)]) > 3:
            std = np.std(self.numberStats[self._canonical(path)])
            if not np.allclose(std, 0.0):
                num /= std

        encoded = self.numberEmbeddings[self._canonical(path)](torch.Tensor([[num]]))
        return encoded


if __name__ == "__main__":

    SOME_JSON = """
    {"menu": {
        "header": "SVG Viewer",
        "id": 23.5,
        "items": [
            {"id": "Open"},
            {"id": "OpenNew", "label": "Open New"},
            null,
            {"id": "ZoomIn", "label": "Zoom In"},
            {"id": "ZoomOut", "label": "Zoom Out"},
            {"id": "OriginalView", "label": "Original View"},
            null,
            {"id": "Quality"},
            {"id": "Pause"},
            {"id": "Mute"},
            null,
            {"id": "Find", "label": "Find..."},
            {"id": "FindAgain", "label": "Find Again"},
            {"id": "Copy"},
            {"id": "CopyAgain", "label": "Copy Again"},
            {"id": "CopySVG", "label": "Copy SVG"},
            {"id": "ViewSVG", "label": "View SVG"},
            {"id": "ViewSource", "label": "View Source"},
            {"id": "SaveAs", "label": "Save As"},
            null,
            {"id": "Help"},
            {"id": "About", "label": "About Adobe CVG Viewer..."}
        ]
    }}
    """

    # json_decoded = json.loads(SOME_JSON)
    # print(" \n\n\n ")
    # print(json_decoded)
    # print("\n\n\nStarting Json decoding... ")

    # model = JSONTreeLSTM()
    # x = model(json_decoded, "some json")

    # print("Representation:")
    # print(x.forward())

    import tqdm

    HIDDEN_SIZE = 128
    embedder = JSONTreeLSTM(HIDDEN_SIZE)
    output_layer = torch.nn.Linear(HIDDEN_SIZE * 2, 2)
    model = torch.nn.Sequential(embedder, torch.nn.ReLU(), output_layer)
    criterion = torch.nn.CrossEntropyLoss()

    BATCH_SIZE = 1
    LEARNING_RATE = 0.001

    # need to run a forward pass to initialise parameters for optimizer :(
    with open("../../Datasets/dota_matches/match_{0:06d}.json".format(0)) as f:
        json_decoded = json.load(f)
    model(json_decoded)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    running_loss = 0.0
    running_acc = 0.0

    for i in tqdm.trange(50000):

        if i % BATCH_SIZE == 0:
            optimizer.zero_grad()

        with open("../../Datasets/dota_matches/match_{0:06d}.json".format(i)) as f:
            json_decoded = json.load(f)

        labs = torch.LongTensor([json_decoded['radiant_win']], )
        json_decoded['radiant_win'] = None

        # json_decoded['teamfights'] = None
        # json_decoded['players'] = None
        json_decoded['objectives'] = None
        json_decoded['radiant_gold_adv'] = None
        json_decoded['radiant_xp_adv'] = None
        json_decoded['chat'] = None
        json_decoded['tower_status_radiant'] = None
        json_decoded['tower_status_dire'] = None
        json_decoded['barracks_status_radiant'] = None
        json_decoded['barracks_status_dire'] = None
        inputs = json_decoded['teamfights']
        # print(json_decoded)

        # forward + backward + optimize
        outputs = model(inputs).view(1, -1)
        loss = criterion(outputs, labs) / BATCH_SIZE
        loss.backward()

        if (i + 1) % BATCH_SIZE == 0:
            optimizer.step()

        # print statistics
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labs)
        running_acc += acc.item()

        running_loss += loss.item() * BATCH_SIZE

        if (i + 1) % 100 == 0:  # print every 1000 mini-batches
            tqdm.tqdm.write('[%d, %5d] loss: %.3f, acc: %.3f' %
                            (1, i + 1, running_loss / 100, running_acc))
            running_loss = 0.0
            running_acc = 0.0
