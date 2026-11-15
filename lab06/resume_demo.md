# Task 7: Interrupt and resume demo

This shows how the training script satisfies: *"Interrupt training at epoch 3, resume from checkpoint. Confirm curves continue smoothly."*

## 1) Run for 3 epochs (then stop)
Edit `config.yaml` temporarily:
```yaml
training:
  epochs: 3
  resume_from: ""
```
Run:
```bash
python src/train.py --config config.yaml
```
This produces `results/checkpoint_last.pth`, `results/checkpoint_epoch_3.pth`, and `results/history.json` with epochs 1-3.

## 2) "Interrupt" and resume to finish
Set in `config.yaml`:
```yaml
training:
  epochs: 10
  resume_from: "results/checkpoint_epoch_3.pth"
```
Run again:
```bash
python src/train.py --config config.yaml
```
The script loads model, optimizer, scheduler state, epoch counter, and history, then continues at epoch 4. The saved history/MLflow metrics will show a continuous curve 1..10 (no reset/jump), confirming smooth resumption.

## Why it continues smoothly
- `save_checkpoint` stores: `epoch`, `model_state`, `optimizer_state`, `scheduler_state`, and `history`.
- `train()` restores all of these and sets `start_epoch = ckpt_epoch + 1`.
- Because optimizer momentum/Adam moments and LR scheduler state are restored, the optimization trajectory continues rather than restarting cold.
