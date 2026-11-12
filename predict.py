from nnunetv2.inference.predict_from_raw_data import predict_entry_point
import traceback
if __name__ == '__main__':
    try:
        predict_entry_point()
    except Exception as e:
        print("预测过程中发生错误:")
        print(str(e))
        traceback.print_exc()
