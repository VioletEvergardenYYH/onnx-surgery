import argparse
import numpy as np
import onnxruntime
from surgery import Surgery
import sys




def validate(model_path):
    # input_shape (w, h)
    sur = Surgery(model_path)

    #sur.fix_loop_body()
    sur.fix_Transpose2()
    # sur.fix_AddV2()
    # sur.remove_pad()
    # sur.fix_ScatterNd()
    # sur.fix_MatrixBandPart()
    # sur.able_run()

    out_path = model_path[:-5]
    
    out_path += '_fix.onnx'
    print('fixed onnx save at ', out_path)

    sur.export(out_path)



if __name__ == "__main__":
    print (sys.argv)
    if len(sys.argv) < 2 :
        print('Usage : input_dfsmnv2_onnx_file')
        import pdb; pdb.set_trace()
    input_file = sys.argv[1]

    validate(input_file)

    # if validate('../title_asr_ocr_c_0828/smooth.onnx') is not None:
    #     print("this onnx model seems ok")
    # else:
    #     print("something wrong, please check your onnx model according to the error message...")
