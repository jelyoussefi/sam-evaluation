from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import SamPredictor, sam_model_registry
from collections import defaultdict
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from statistics import mean
from tqdm import tqdm
from torch.nn.functional import threshold, normalize


bbox_coords = {}
for f in sorted(Path('ground-truth-maps/ground-truth-maps/').iterdir())[:100]:
	k = f.stem[:-3]
	if k not in stamps_to_exclude:
		im = cv2.imread(f.as_posix())
		gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
		if len(contours) > 1:
			x,y,w,h = cv2.boundingRect(contours[0])
			height, width, _ = im.shape
			bbox_coords[k] = np.array([x, y, x + w, y + h])

ground_truth_masks = {}
for k in bbox_coords.keys():
	gt_grayscale = cv2.imread(f'ground-truth-pixel/ground-truth-pixel/{k}-px.png', cv2.IMREAD_GRAYSCALE)
	ground_truth_masks[k] = (gt_grayscale == 0)
	

def show_mask(mask, ax, random_color=False):
	if random_color:
		color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
	else:
		color = np.array([30/255, 144/255, 255/255, 0.6])
	h, w = mask.shape[-2:]
	mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
	ax.imshow(mask_image)
    
def show_box(box, ax):
	x0, y0 = box[0], box[1]
	w, h = box[2] - box[0], box[3] - box[1]
	ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

model_type = 'vit_b'
checkpoint = 'sam_vit_b_01ec64.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model.to(device)
sam_model.train();


transformed_data = defaultdict(dict)
for k in bbox_coords.keys():
	image = cv2.imread(f'scans/scans/{k}.png')
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
	transform = ResizeLongestSide(sam_model.image_encoder.img_size)
	input_image = transform.apply_image(image)
	input_image_torch = torch.as_tensor(input_image, device=device)
	transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

	input_image = sam_model.preprocess(transformed_image)
	original_image_size = image.shape[:2]
	input_size = tuple(transformed_image.shape[-2:])

	transformed_data[k]['image'] = input_image
	transformed_data[k]['input_size'] = input_size
	transformed_data[k]['original_image_size'] = original_image_size
	
lr = 1e-4
wd = 0
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)

loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.BCELoss()
keys = list(bbox_coords.keys())

num_epochs = 100
losses = []

for epoch in range(num_epochs):
	epoch_losses = []
	# Just train on the first 20 examples
	for k in keys[:20]:
		input_image = transformed_data[k]['image'].to(device)
		input_size = transformed_data[k]['input_size']
		original_image_size = transformed_data[k]['original_image_size']
    
		# No grad here as we don't want to optimise the encoders
		with torch.no_grad():
			image_embedding = sam_model.image_encoder(input_image)

			prompt_box = bbox_coords[k]
			box = transform.apply_boxes(prompt_box, original_image_size)
			box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
			box_torch = box_torch[None, :]

			sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
											  points=None,
											  boxes=box_torch,
											  masks=None,
											)
		low_res_masks, iou_predictions = sam_model.mask_decoder(
									image_embeddings=image_embedding,
									image_pe=sam_model.prompt_encoder.get_dense_pe(),
									sparse_prompt_embeddings=sparse_embeddings,
									dense_prompt_embeddings=dense_embeddings,
									multimask_output=False,
									)

		upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
		binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

		gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (1, 1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).to(device)
		gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

		loss = loss_fn(binary_mask, gt_binary_mask)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		epoch_losses.append(loss.item())
	losses.append(epoch_losses)
	print(f'EPOCH: {epoch}')
	print(f'Mean loss: {mean(epoch_losses)}')

torch.save(sam_model.state_dict(), "./models/best.pth")
