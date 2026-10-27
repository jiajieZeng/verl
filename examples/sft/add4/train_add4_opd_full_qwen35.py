"""Qwen3.5-9B full-vocabulary OPD; original SFT/top-128 entries remain separate."""
from train_add4_opd import main

if __name__ == '__main__':
    main(expected_student_type='qwen3_5', expected_loss_type='full_vocab_reverse_kl')
