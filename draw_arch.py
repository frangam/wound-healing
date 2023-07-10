#!venv/bin/python3

import graphviz

dot = graphviz.Digraph()

dot.node("input_1", "Input (Features)")
dot.node("input_2", "Input (Images)")
dot.node("lstm", "LSTM")
dot.node("conv2d", "Conv2D")
dot.node("batch_normalization", "Batch Normalization")
dot.node("conv_lstm2d", "ConvLSTM2D")
dot.node("time_distributed", "Time Distributed")
dot.node("time_distributed_1", "Time Distributed")
dot.node("concatenate", "Concatenate")
dot.node("time_distributed_2", "Time Distributed")

dot.edge("input_1", "lstm")
dot.edge("input_2", "conv2d")
dot.edge("conv2d", "batch_normalization")
dot.edge("lstm", "time_distributed")
dot.edge("conv2d", "conv_lstm2d")
dot.edge("batch_normalization", "time_distributed")
dot.edge("conv_lstm2d", "time_distributed_1")
dot.edge("time_distributed", "concatenate")
dot.edge("time_distributed_1", "concatenate")
dot.edge("concatenate", "time_distributed_2")

dot.render("architecture", format="png")
