import os
import argparse
import datetime
import paddle.fluid as fluid
from visualdl import LogWriter

from data_loader import get_token2id_dict, train_loader, val_test_loader
from induction_networks import InductionNetworks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str,
                        help="Path of train data.")
    parser.add_argument("--val_data_path", type=str,
                        help="Path of val data.")
    parser.add_argument("--test_data_path", type=str,
                        help="Path of test data.")
    parser.add_argument("-N", default=2, type=int, help="N way")
    parser.add_argument("-K", default=5, type=int, help="K shot.")
    parser.add_argument("-Q", default=5, type=int,
                        help="Number of query instances per class.")
    parser.add_argument("--train_episodes", default=10000, type=int,
                        help="Number of training episodes. (train_episodes*=batch_size)")
    parser.add_argument("--val_steps", default=100, type=int,
                        help="Validate after x train_episodes.")
    parser.add_argument("--max_length", default=512, type=int,
                        help="Maximum length of sentences.")
    parser.add_argument("--hidden_size", default=128, type=int, help="Hidden size.")
    parser.add_argument("--att_dim", default=64, type=int,
                        help="Attention dimension of Self-Attention Bi-LSTM encoder.")
    parser.add_argument("--induction_iters", default=3, type=int,
                        help="Number of induction iters.")
    parser.add_argument("--relation_size", default=100, type=int,
                        help="Size of neural tensor network.")
    parser.add_argument("-B", "--batch_size", default=4, type=int, help="Batch size.")
    parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate.")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--emb_path", default="./resources/",
                        type=str, help="Path of embedding file.")
    parser.add_argument("--logdir", default="./log", type=str,
                        help="Path of visualdl log.")
    args = parser.parse_args()
    print("Args:", args)
    
    model = InductionNetworks(args.emb_path, args.N, args.K, args.max_length,
                              args.hidden_size, args.att_dim, args.induction_iters,
                              args.relation_size)
    
    train(model, args)
    

def train(model, args):
    # 1. Create VisualDL logger
    logwriter = LogWriter(os.path.join(args.logdir, "visualdl_log"), sync_cycle=10)
    with logwriter.mode("Train") as writer:
        train_loss_scalar = writer.scalar("loss")
        train_acc_scalar = writer.scalar("acc")
        histogram1 = writer.histogram("Relation-BiLinear-W", 100)
        histogram2 = writer.histogram("Relation-BiLinear-b", 10)
        histogram3 = writer.histogram("Relation-FC-W", 100)
    with logwriter.mode("Val") as writer:
        val_acc_scalar = writer.scalar("acc")

    # 2. Setup program
    train_prog = fluid.default_main_program()
    train_startup = fluid.default_startup_program()

    train_reader = model.train_reader
    val_reader = model.val_reader
    test_reader = model.test_reader
    loss = model.loss
    mean_acc = model.mean_acc

    # Clone for val / test
    val_prog = train_prog.clone(for_test=True)
    test_prog = train_prog.clone(for_test=True)

    optimizer = fluid.optimizer.Adam(learning_rate=args.lr)
    optimizer.minimize(loss)

    # 3. Setup executor
    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(train_startup)

    # 4. Get Relation Module params for VisualDL
    # print(fluid.io.get_program_parameter(train_startup))
    relation_BL_w = train_startup.global_block().var("Relation-BiLinear.w_0")
    relation_BL_b = train_startup.global_block().var("Relation-BiLinear.b_0")
    relation_FC_w = train_startup.global_block().var("Relation-FC.w_0")

    # 5. Compile
    print("Compilling...")
    compiled_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(loss_name=loss.name)
    compiled_val_prog = fluid.CompiledProgram(val_prog).with_data_parallel(share_vars_from=compiled_train_prog)
    compiled_test_prog = fluid.CompiledProgram(test_prog).with_data_parallel(share_vars_from=compiled_train_prog)

    # 6. Setup data source
    token2idx_dict, unk_idx, pad_idx = get_token2id_dict(args.emb_path)
    print("Setup dataloader...")
    places = fluid.cuda_places() if args.use_cuda else fluid.cpu_places()
    train_reader.set_sample_generator(
        train_loader(args.train_data_path, args.N, args.K, args.Q, token2idx_dict,
                     unk_idx, pad_idx, args.max_length),
        batch_size=args.batch_size, places=places)
    val_reader.set_sample_generator(
        val_test_loader(args.val_data_path, args.N, args.K, args.Q, token2idx_dict,
                        unk_idx, pad_idx, args.max_length, data_type="val"),
        batch_size=1, places=places)
    test_reader.set_sample_generator(
        val_test_loader(args.test_data_path, args.N, args.K, args.Q, token2idx_dict,
                        unk_idx, pad_idx, args.max_length, data_type="test"),
        batch_size=1, places=places)

    # 7. Train loop
    # For save best model
    best_val_acc = 0
    # Count result for each args.val_steps
    total_sample = total_loss = total_acc = 0
    for epi, train_data in zip(range(1, args.train_episodes + 1), train_reader()):
        # 7.1 Run
        (train_cur_loss, train_cur_acc, relation_BL_w_value,
        relation_BL_b_value, relation_FC_w_value) = exe.run(
            program=compiled_train_prog, feed=train_data,
            fetch_list=[loss.name, mean_acc.name,
                        relation_BL_w.name, relation_BL_b.name,
                        relation_FC_w.name])
        total_loss += train_cur_loss[0]
        total_acc += train_cur_acc[0]
        total_sample += 1
        # print(train_cur_loss, train_cur_acc)
        # print(total_loss, total_acc, total_sample)

        if epi % 10 == 0:
            print("{}  [Train episode: {:5d}/{:5d}] ==> Loss: {:2.4f} Mean acc: {:2.4f}"
                  .format(str(datetime.datetime.now())[:-7], epi, args.train_episodes,
                  total_loss / total_sample, 100 * total_acc / total_sample))

        # 7.2 Add params histogram to VisualDL
        histogram1.add_record(epi, relation_BL_w_value.flatten())
        histogram2.add_record(epi, relation_BL_b_value.flatten())
        histogram3.add_record(epi, relation_FC_w_value.flatten())

        # 7.3 Validation
        if args.val_data_path and epi % args.val_steps == 0:            
            # 7.3.1 Add train loss/acc to VisualDL
            train_loss_scalar.add_record(epi, total_loss / total_sample)
            train_acc_scalar.add_record(epi, total_acc / total_sample)
            total_sample = total_loss = total_acc = 0

            # 7.3.2 Run val once
            val_acc_mean = eval(
                exe, compiled_val_prog, val_reader, [mean_acc.name], run_type="Val")

            print("{}  [Val result: {:5d}/{:5d}] ==> Mean acc: {:2.4f}"
                  .format(str(datetime.datetime.now())[:-7], epi,
                          args.train_episodes, 100 * val_acc_mean))
            # Add val acc to VisualDL
            val_acc_scalar.add_record(epi, val_acc_mean)

            # 7.3.3 Save best model
            if val_acc_mean > best_val_acc:
                best_val_acc = val_acc_mean
                fluid.io.save_inference_model(
                    os.path.join(args.logdir, "infer_model"),
                    ["totalQ", "support", "support_len", "query", "query_len"],
                    [model.prediction], exe, main_program=train_prog,
                    params_filename="__params__")
                print("{}  [Save model of val mean acc: {:2.4f}] ==> {}"
                      .format(str(datetime.datetime.now())[:-7], 100 * best_val_acc,
                      os.path.join(args.logdir, "infer_model")))

    # 8. Test
    if args.test_data_path:
        test_acc_mean = eval(exe, compiled_test_prog, test_reader, [mean_acc.name], run_type="Test")
        print("{}  [Test result] ==> Mean acc: {:2.4f}"
            .format(str(datetime.datetime.now())[:-7], 100 * test_acc_mean))


def eval(exe, program, data_reader, fetch_list, run_type="Val"):
    total_eval_sample = total_eval_acc = 0
    for eval_data in data_reader():
        eval_cur_acc, = exe.run(program=program, feed=eval_data,
                               fetch_list=fetch_list)
        total_eval_acc += eval_cur_acc[0]
        total_eval_sample += 1

        if total_eval_sample % 100 == 0:
            print("{}  [{} step: {:4d}] ==> Mean acc: {:2.4f}"
                  .format(str(datetime.datetime.now())[:-7], run_type,
                          total_eval_sample,
                          100 * total_eval_acc / total_eval_sample))

    # Results for this eval
    _eval_acc_mean = total_eval_acc / total_eval_sample
    return _eval_acc_mean


if __name__ == "__main__":
    main()