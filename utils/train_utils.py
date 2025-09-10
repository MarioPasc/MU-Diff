import os
import json
import math
from typing import Dict, Any, List, Optional

import torch
import torchvision
from torchvision.utils import make_grid
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import numpy as np

def _ensure_dir(p: str):
	os.makedirs(p, exist_ok=True)

def _to_cpu_float(t: torch.Tensor) -> torch.Tensor:
	if t is None:
		return t
	return t.detach().to('cpu', dtype=torch.float32)

def build_real_fake_collage(real: torch.Tensor, fake: torch.Tensor, max_rows: int = 4) -> torch.Tensor:
	"""
	Build a 4x4 style collage: columns 0 & 2 real, 1 & 3 fake.
	real/fake: [B, C, H, W] in [-1,1] or [0,1]. We'll normalize to [0,1].
	"""
	if real.ndim != 4 or fake.ndim != 4:
		raise ValueError("real/fake must be BCHW")
	b = min(real.size(0), max_rows*2)  # need pairs
	if b < 2:
		# pad by repeating
		real = real.repeat(2,1,1,1)
		fake = fake.repeat(2,1,1,1)
		b = 2
	# pick first b samples
	real = real[:b]
	fake = fake[:b]
	# scale to [0,1]
	def norm(x):
		if x.min() < -0.01:
			x = (x + 1)/2
		return x.clamp(0,1)
	real = norm(real)
	fake = norm(fake)
	rows = min(max_rows, b//2)
	# Build list of images row-wise: [real_i, fake_i, real_j, fake_j]
	imgs: List[torch.Tensor] = []
	for r in range(rows):
		i = 2*r
		j = 2*r+1
		imgs.extend([real[i], fake[i], real[j], fake[j]])
	grid = make_grid(imgs, nrow=4, padding=2)
	return grid

def save_collage_with_labels(collage: torch.Tensor, path: str):
	_ensure_dir(os.path.dirname(path))
	# Convert to numpy for labeling overlay using matplotlib
	np_img = collage.cpu().numpy()
	if np_img.shape[0] in (1,3):
		# C,H,W -> H,W,C
		np_img = np.transpose(np_img, (1,2,0))
	plt.figure(figsize=(6,6))
	plt.imshow(np_img.squeeze(), cmap='gray' if np_img.ndim==2 or np_img.shape[2]==1 else None)
	# Add column labels at top
	labels = ["Real A", "Fake A", "Real B", "Fake B"]
	w = np_img.shape[1]
	cell_w = w / 4
	for ci, lab in enumerate(labels):
		plt.text(ci*cell_w + cell_w/2, 10, lab, color='yellow', ha='center', va='top', fontsize=8, bbox=dict(facecolor='black', alpha=0.4, pad=2))
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(path, dpi=150)
	plt.close()

def update_history(history_path: str, record: Dict[str, Any]):
	data: List[Dict[str, Any]] = []
	if os.path.isfile(history_path):
		try:
			with open(history_path, 'r') as f:
				data = json.load(f)
		except Exception:
			data = []
	data.append(record)
	with open(history_path, 'w') as f:
		json.dump(data, f, indent=2)

def plot_history(history_path: str, out_dir: str):
	if not os.path.isfile(history_path):
		return
	with open(history_path,'r') as f:
		hist = json.load(f)
	if not hist:
		return
	_ensure_dir(out_dir)
	# Extract metrics arrays
	epochs = [r['epoch'] for r in hist]
	def _maybe(metric):
		return [r.get(metric, float('nan')) for r in hist]
	# Loss plot
	plt.figure(figsize=(8,5))
	for k in ['train_G_total','train_D_total','val_l1','val_psnr']:
		if any(not math.isnan(v) for v in _maybe(k)):
			plt.plot(epochs, _maybe(k), label=k)
	plt.xlabel('Epoch'); plt.ylabel('Value'); plt.title('Loss / PSNR'); plt.legend(); plt.grid(True, alpha=0.3)
	plt.tight_layout(); plt.savefig(os.path.join(out_dir,'loss_psnr_evolution.png'), dpi=150); plt.close()

	# Time + memory
	plt.figure(figsize=(8,4))
	for k in ['epoch_time_sec','peak_mem_mb']:
		if any(not math.isnan(v) for v in _maybe(k)):
			plt.plot(epochs, _maybe(k), label=k)
	plt.xlabel('Epoch'); plt.ylabel('Time (s) / Mem (MB)'); plt.title('Resource Usage'); plt.legend(); plt.grid(True, alpha=0.3)
	plt.tight_layout(); plt.savefig(os.path.join(out_dir,'resource_evolution.png'), dpi=150); plt.close()

def epoch_visual_report(
	out_dir: str,
	epoch: int,
	real_batch: torch.Tensor,
	fake_batch: torch.Tensor,
	avg_losses: Dict[str, float],
	val_metrics: Dict[str, float],
	epoch_time_sec: float,
	peak_mem_mb: float,
	extra: Optional[Dict[str, Any]] = None,
):
	"""Create collage, update JSON history, and plot evolution files.
	Parameters mimic what's available at epoch end.
	"""
	reports_dir = os.path.join(out_dir, 'epoch_reports')
	_ensure_dir(reports_dir)

	# Collage
	try:
		collage = build_real_fake_collage(real_batch, fake_batch)
		collage_path = os.path.join(reports_dir, f'collage_epoch_{epoch:04d}.png')
		save_collage_with_labels(collage, collage_path)
	except Exception as e:
		print(f"[REPORT] Collage failed epoch {epoch}: {e}")

	# Prepare history record
	rec: Dict[str, Any] = {
		'epoch': epoch,
		'train_G_total': avg_losses.get('G_total', float('nan')),
		'train_D_total': avg_losses.get('D_total', float('nan')),
		'train_G_adv': avg_losses.get('G_adv', float('nan')),
		'train_G_L1': avg_losses.get('G_L1', float('nan')),
		'train_G_mask': avg_losses.get('G_mask', float('nan')),
		'train_D_real': avg_losses.get('D_real', float('nan')),
		'train_D_fake': avg_losses.get('D_fake', float('nan')),
		'train_R1': avg_losses.get('R1', float('nan')),
		'val_psnr': val_metrics.get('val_psnr', float('nan')) if val_metrics else float('nan'),
		'val_l1': val_metrics.get('val_l1', float('nan')) if val_metrics else float('nan'),
		'epoch_time_sec': epoch_time_sec,
		'peak_mem_mb': peak_mem_mb,
	}
	if extra:
		rec.update(extra)
	history_path = os.path.join(out_dir, 'training_history.json')
	update_history(history_path, rec)
	# Refresh plots
	try:
		plot_history(history_path, reports_dir)
	except Exception as e:
		print(f"[REPORT] Plot failed epoch {epoch}: {e}")

	return rec

