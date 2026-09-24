# MR-HHGNN

PyTorch implementation of the hierarchical aggregation modules in

> **A novel Hierarchical Heterogeneous Graph Network with Multi-domain Representation for Remaining Useful Life Prediction of Bearings**
> Chenyang Hou, Xiaoguang Zhang\*, Haiyu Guo, Weiming Shen, Mingjian Zuo, Yiyang Liu, Peng Gao, Qiao Yu

## Usage

```bash
pip install -r requirements.txt
python demo.py          # forward + backward pass on synthetic inputs
```

PyTorch is the only dependency; a GPU is not required for the demo.

## Structure

| File | Content | Paper |
|---|---|---|
| `mrhhgnn/relations.py` | relation schema of the MDHG (12 typed relations) | §3.2 |
| `mrhhgnn/node_level.py` | relation-aware node-level aggregation | §3.3.1 |
| `mrhhgnn/hierarchical.py` | path-level and cross-domain aggregation, stacked layers | §3.3.2–3.3.3 |

## License

MIT
