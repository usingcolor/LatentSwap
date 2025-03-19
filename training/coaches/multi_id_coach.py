import os
import copy

import torch
# from tqdm import tqdm
from training.projectors import w_projector

from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach


class MultiIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb=False):
        
        super().__init__(data_loader, use_wandb)
        device = torch.device(global_config.device)
        self.G.to(device)
        self.G_copy = copy.deepcopy(self.G).eval().requires_grad_(False).to(device).float()  # type: ignore


    def train(self):
        self.G.synthesis.train()
        self.G.mapping.train()

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{global_config.run_name}', exist_ok=True)

        use_ball_holder = True
        w_pivots = []
        images = []

        for i, (fname, image) in enumerate(self.data_loader):
            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            image_name = fname[0]
            if hyperparameters.first_inv_type == 'w+':
                embedding_dir = f'{w_path_dir}/{global_config.run_name}/{image_name}'
            else:
                embedding_dir = f'{w_path_dir}/{global_config.run_name}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = self.get_inversion(w_path_dir, image_name, image)
            w_pivots.append(w_pivot)
            images.append((image_name, image))
            # torch.save(w_pivot, f'{embedding_dir}/0.pt')
            self.image_counter += 1

#        for i in tqdm(range(hyperparameters.max_pti_steps)):
        for i in range(hyperparameters.max_pti_steps):
            self.image_counter = 0

            for data, w_pivot in zip(images, w_pivots):
                image_name, image = data

                if self.image_counter >= hyperparameters.max_images_to_invert:
                    break

                real_images_batch = image.to(global_config.device)

                generated_images = self.forward(w_pivot)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                                      self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                global_config.training_step += 1
                self.image_counter += 1

        # torch.save(self.G,
        #            f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_multi_id.pt')

        return self.G, w_pivots
    
    
    def ff_train(self, image_tensor_list):
        self.G.synthesis.train()
        self.G.mapping.train()

        # w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        # os.makedirs(w_path_dir, exist_ok=True)
        # os.makedirs(f'{w_path_dir}/{global_config.run_name}', exist_ok=True)

        use_ball_holder = True
        w_pivots = []
        images = []

        for i, (image) in enumerate(image_tensor_list):
            if self.image_counter >= hyperparameters.max_images_to_invert:
                break
            
            id_image = torch.squeeze((image + 1) / 2) * 255
            w_pivot = w_projector.project(self.G, id_image, device=torch.device(global_config.device), w_avg_samples=600,
                                    num_steps=hyperparameters.first_inv_steps, w_name='',)
            w_pivots.append(w_pivot)
            
        for i in range(hyperparameters.max_pti_steps):
            for image, w_pivot in zip(image_tensor_list, w_pivots):
                generated_images = self.forward(w_pivot)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, image, '',
                                      self.G, use_ball_holder, w_pivot)
                self.optimizer.zero_grad()
                # loss.requires_grad_(True)
                loss.backward()
                self.optimizer.step()
        return self.G, w_pivots
    
    def reset(self):
        del self.G
        del self.optimizer
        self.G = copy.deepcopy(self.G_copy)
        self.optimizer = self.configure_optimizers()
