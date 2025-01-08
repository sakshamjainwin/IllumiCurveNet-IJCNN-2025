import os
import argparse
import torch
import torch.optim
import model
import utils.losses as losses
import utils.dataloader as dataloader

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)
          
def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    DCE_net = model.illumi_curve_net().cuda()

    DCE_net.apply(weights_init)
    if config.load_pretrain == True:
        DCE_net.load_state_dict(torch.load(config.pretrain_dir))

    # Define the loss configurations
    loss_configs = [
        {"name": "dynamic_exposure", 
         "weights": {"L_color": 5.0, "L_spa": 1.5, "L_exp_dynamic": 7.0, "L_TV": 200.0, "L_contrast": 5.0, "L_exp": 3.0, "L_texture": 3.0}}
    ]

    # Loop over each loss configuration
    for loss_config in loss_configs:
        print(f"Training with loss configuration: {loss_config['name']}")

        # Model initialization
        DCE_net = model.enhance_net_nopool().cuda()
        DCE_net.apply(weights_init)

        if config["load_pretrain"]:
            DCE_net.load_state_dict(torch.load(config["pretrain_dir"]))

        # Load dataset
        train_dataset = dataloader.lowlight_loader(config["lowlight_images_path"])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=True,
                                                   num_workers=config["num_workers"], pin_memory=True)

        # Initialize losses
        L_color = losses.L_color()
        L_spa = losses.L_spa()
        L_texture = losses.L_texture()
        L_exp_dynamic = losses.L_exp_dynamic()
        L_contrast = losses.L_contrast()
        L_TV = losses.L_TV()

        # Set exposure value for exposure loss
        E = 0.6
        L_exp = losses.L_exp(16, E)

        # Optimizer
        optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        DCE_net.train()

        # Early stopping parameters
        best_loss = float('inf')
        patience = config.early_stopping_patience
        patience_counter = 0
        min_delta = 1e-4

        for epoch in range(config["num_epochs"]):
            epoch_loss = 0.0
            num_batches = 0

            for iteration, img_lowlight in enumerate(train_loader):
                img_lowlight = img_lowlight.cuda()

                # Forward pass
                enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)

                # Compute loss dynamically based on configuration
                loss = 0.0
                weights = loss_config["weights"]

                if "L_color" in weights:
                    loss += weights["L_color"] * torch.mean(L_color(enhanced_image))
                if "L_spa" in weights:
                    loss += weights["L_spa"] * torch.mean(L_spa(enhanced_image, img_lowlight))
                if "L_exp" in weights:
                    loss += weights["L_exp"] * torch.mean(L_exp(enhanced_image))
                if "L_TV" in weights:
                    loss += weights["L_TV"] * torch.mean(L_TV(A))
                if "L_texture" in weights:
                    loss += weights["L_texture"] * torch.mean(L_texture(img_lowlight, enhanced_image))
                if "L_exp_dynamic" in weights:
                    loss += weights["L_exp_dynamic"] * torch.mean(L_exp_dynamic(enhanced_image, img_lowlight))
                if "L_contrast" in weights:
                    loss += weights["L_contrast"] * torch.mean(L_contrast(enhanced_image))

                if torch.isnan(loss).any():
                    print("NaN detected in loss. Skipping iteration.")
                    continue  # Skip this iteration

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), config["grad_clip_norm"])
                torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # Display progress
                if (iteration + 1) % config["display_iter"] == 0:
                    print(f"Epoch [{epoch+1}/{config['num_epochs']}], Iteration [{iteration+1}], Loss: {loss.item()}")

                # Save checkpoint
                if (iteration + 1) % config["snapshot_iter"] == 0:
                    checkpoint_name = f"model-{loss_config['name']}-epoch{epoch}-iteration{iteration}.pth"
                    torch.save(DCE_net.state_dict(), os.path.join(config["snapshots_folder"], checkpoint_name))

            # Calculate average loss for the epoch
            avg_epoch_loss = epoch_loss / num_batches

            # Early stopping check
            if avg_epoch_loss < best_loss - min_delta:
                best_loss = avg_epoch_loss
                patience_counter = 0
                # Save best model
                best_model_name = f"model-best.pth"
                torch.save(DCE_net.state_dict(), os.path.join(config["snapshots_folder"], best_model_name))
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs \n")
                break

        print(f"Finished training with loss configuration: {loss_config['name']}")

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Low-light Image Enhancement Training Script')

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/", help='Path to low-light training images')
	parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
	parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for optimizer')
	parser.add_argument('--grad_clip_norm', type=float, default=0.1, help='Gradient clipping norm')
	parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
	parser.add_argument('--train_batch_size', type=int, default=8, help='Training batch size')
	parser.add_argument('--val_batch_size', type=int, default=4, help='Validation batch size')
	parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
	parser.add_argument('--display_iter', type=int, default=10, help='Display loss every N iterations')
	parser.add_argument('--snapshot_iter', type=int, default=10, help='Save model every N iterations')
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/", help='Directory to save model checkpoints')
	parser.add_argument('--load_pretrain', type=bool, default= False, help='Whether to load pretrained model')
	parser.add_argument('--early_stopping_patience', type=int, default=10, help='Number of epochs to wait before early stopping')
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/model-best.pth")

	config = parser.parse_args()

	# Create snapshots directory if it doesn't exist
	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)

	train(config)