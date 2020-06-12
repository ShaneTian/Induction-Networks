import argparse
import numpy as np
from paddle import fluid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        help="Path of __model__ and __params__")
    parser.add_argument("--use_cuda", action="store_true")
    args = parser.parse_args()
    print("Args:", args)

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # Load inference model
    inference_program, feed_target_names, fetch_targets = load_model(args.model_path, exe)
    print("Feed target names:", feed_target_names)
    print("Fetch targets:", fetch_targets)
    
    # A temp sample
    B, N, K, Q = 4, 2, 5, 5
    max_length = 512
    totalQ = np.array([N * Q], dtype=np.int32)
    support = np.random.randint(0, high=1000, size=[B, N, K, max_length])
    support_len = np.random.randint(10, high=max_length, size=[B, N, K])
    query = np.random.randint(0, high=1000, size=[B, N * Q, max_length])
    query_len = np.random.randint(10, high=max_length, size=[B, N * Q])

    # Run inference model
    pred_label, = exe.run(inference_program,
          feed={
              feed_target_names[0]: totalQ,
              feed_target_names[1]: support,
              feed_target_names[2]: support_len,
              feed_target_names[3]: query,
              feed_target_names[4]: query_len
          },
          fetch_list=fetch_targets)
    print("The predict label is:", pred_label)  # [B, totalQ]


def load_model(model_path, exe):
    [inference_program, feed_target_names, fetch_targets] = (
        fluid.io.load_inference_model(dirname=model_path, executor=exe,
                                      params_filename="__params__"))
    return inference_program, feed_target_names, fetch_targets


if __name__ == "__main__":
    main()