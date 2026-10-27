"""Qwen3.6-35B-A3B full-vocabulary OPD using the frozen teacher on 204."""
from train_add4_opd import main

if __name__ == '__main__':
    main(expected_student_type='qwen3_5_moe', expected_loss_type='full_vocab_reverse_kl')
