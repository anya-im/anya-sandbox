import torch.onnx
import torch
from anyasand.model import AnyaAE
from anyasand.dictionary import DictionaryTrainer


def ConvertONNX():
    # set the model to inference mode
    dicts = DictionaryTrainer("./anya-dic.db")
    model = AnyaAE(dicts.input_vec_size)
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn((1, dicts.input_vec_size,))

    # Export the model
    torch.onnx.export(model,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         "anya.onnx",       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         opset_version=10,    # the ONNX version to export the model to
         do_constant_folding=True,  # whether to execute constant folding for optimization
         input_names = ['modelInput'],   # the model's input names
         output_names = ['modelOutput'], # the model's output names
         dynamic_axes={'modelInput' : {0 : 'batch_size'}, 'modelOutput' : {0 : 'batch_size'}})

    print(" ")
    print('Model has been converted to ONNX')


def main():
    ConvertONNX()


if __name__ == "__main__":
    main()

