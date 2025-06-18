# run_a_trial()
from optimizer import *

if __name__ == "__main__":
    while True:
        tf.keras.backend.clear_session() # limpar memoria a cada execucao
        # Optimize a new model with the TPE Algorithm:
        print("OPTIMIZING NEW MODEL:")
        try:
            run_a_trial()
        except Exception as err:
            err_str = str(err)
            print(err_str)
            #traceback_str = str(traceback.format_exc())
            #print(traceback_str)