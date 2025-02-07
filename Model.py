import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4, padding=1)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, padding=1)
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(16 * 7, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)  # Latent space
        
        # Decoder (Pathway 1)
        self.fc4 = nn.Linear(16, 32)
        self.fc5 = nn.Linear(32, 64)
        self.fc6 = nn.Linear(64, 16 * 7)
        self.deconv1 = nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=4, stride=2)
        self.deconv2 = nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=4, stride=2)
        self.deconv3 = nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
        self.deconv4 = nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, stride=2)

        # Pathway 2: MLP for simulation type classification
        self.fc_sim1 = nn.Linear(16, 100)
        self.fc_sim2 = nn.Linear(100, 100)
        self.fc_sim3 = nn.Linear(100, 10)
        
        # Pathway 3: MLP for SM and SFR prediction
        self.fc_sfr1 = nn.Linear(16, 200)
        self.fc_sfr2 = nn.Linear(200, 200)
        self.fc_sfr3 = nn.Linear(200, 200) 
        self.fc_sfr4 = nn.Linear(200, 2)  # Predicting SM and SFR

    def encoder(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.view(-1, 16 * 7)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        encoded = self.fc3(x)
        return encoded

    def decoder(self, x):
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = x.view(-1, 16, 7)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv3(x))
        x = torch.relu(self.deconv4(x))
        return x

    def sim_type_classifier(self, x):
        x = torch.relu(self.fc_sim1(x))
        x = torch.relu(self.fc_sim2(x))
        sim_type_output = self.fc_sim3(x)
        return sim_type_output

    def sfr_predictor(self, x):
        x = torch.relu(self.fc_sfr1(x))
        x = torch.relu(self.fc_sfr2(x))
        x = torch.relu(self.fc_sfr3(x))
        sfr_output = self.fc_sfr4(x)
        return sfr_output
    
    def forward(self, x):
        latent = self.encoder(x)
        
        # Pathway 1: Reconstruct SFH
        sfh_output = self.decoder(latent)
        
        # Pathway 2: Classify sim type
        sim_type_output = self.sim_type_classifier(latent)
        
        # Pathway 3: Predict SM and SFR
        sfr_output = self.sfr_predictor(latent)
        
        return sfh_output, sim_type_output, sfr_output

def compute_loss(sfh_output, sfh_target, sim_type_output, sim_type_target, mass_sfr_output, mass_sfr_target, w_reg, w_cl):
    mse_loss = nn.MSELoss()
    cross_entropy_loss = nn.CrossEntropyLoss()

    loss_sfh = mse_loss(sfh_output, sfh_target)
    loss_sim_type = cross_entropy_loss(sim_type_output, sim_type_target)
    loss_mass_sfr = mse_loss(mass_sfr_output, mass_sfr_target)

    total_loss = loss_sfh + w_reg * loss_mass_sfr + w_cl * loss_sim_type
    # print(f'Loss: {total_loss:.3f} | SFH Loss: {loss_sfh:.3f} | Sim Type Loss: {loss_sim_type:.3f} | Mass-SFR Loss: {loss_mass_sfr:.3f}')
    return total_loss, loss_sfh, loss_sim_type, loss_mass_sfr

def train(AE, num_epochs, device, optimizer, scheduler, train_loader, val_loader):
        """
        Training the model AE with specified configuration; return a list of training losses, and validation losses for mass_sfr, sfh, sim_type.
        """
        losses = [] # To store training losses
        losses_sfh = []  # To store SFH losses
        losses_sim_type = []  # To store sim type losses
        losses_mass_sfr = []  # To store mass-sfr losses
        val_losses = []  # To store validation losses
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            for data in train_loader:
                inputs, sim_labels, mass_sfr = data
                optimizer.zero_grad()
                inputs = inputs.to(device)
                sim_labels = sim_labels.to(device)
                mass_sfr = mass_sfr.to(device)
                sfh_output, sim_type_output, mass_sfr_output = AE(inputs.unsqueeze(1))
                sfh_output = sfh_output.squeeze(1)
                loss = compute_loss(sfh_output, inputs, sim_type_output, sim_labels, mass_sfr_output, mass_sfr, 1, 1)
                loss = loss[0]
                loss.backward()
                optimizer.step()
                scheduler.step() # Experimenting with cyclicLR, updates per *batch*
                
                total_loss += loss.item()

            total_loss /= len(train_loader)
            losses.append(total_loss)
            
            # Validation 
            AE.eval()
            val_loss = 0.0
            sfh_loss = 0.0
            sim_type_loss = 0.0
            mass_sfr_loss = 0.0
            with torch.no_grad():
                for data in val_loader:
                    inputs, sim_labels, mass_sfr = data
                    inputs = inputs.to(device)
                    sim_labels = sim_labels.to(device)
                    mass_sfr = mass_sfr.to(device)
                    sfh_output, sim_type_output, mass_sfr_output = AE(inputs.unsqueeze(1))
                    loss, loss_sfh, loss_sim_type, loss_mass_sfr = compute_loss(sfh_output, inputs, sim_type_output, sim_labels, mass_sfr_output, mass_sfr, 1, 1)
                    val_loss += loss.item()
                    sfh_loss += loss_sfh.item()
                    sim_type_loss += loss_sim_type.item()
                    mass_sfr_loss += loss_mass_sfr.item()
                    
            
            val_loss /= len(val_loader)
            sfh_loss /= len(val_loader)
            sim_type_loss /= len(val_loader)
            mass_sfr_loss /= len(val_loader)
            val_losses.append(val_loss)
            losses_sfh.append(sfh_loss)
            losses_sim_type.append(sim_type_loss)
            losses_mass_sfr.append(mass_sfr_loss)
            
            print(f'Epoch [{epoch}/{num_epochs}], Training Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, SFH Loss: {sfh_loss:.4f}, Sim Type Loss: {sim_type_loss:.4f}, Mass-SFR Loss: {mass_sfr_loss:.4f}')

        print('Training complete')
        return losses, losses_mass_sfr, losses_sfh, losses_sim_type

def train_no_val(AE, num_epochs, device, optimizer, scheduler, train_loader):
        """
        Training the model AE with specified configuration; return a list of training losses.
        """
        losses = [] # To store training losses
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            for data in train_loader:
                inputs, sim_labels, mass_sfr = data
                optimizer.zero_grad()
                inputs = inputs.to(device)
                sim_labels = sim_labels.to(device)
                mass_sfr = mass_sfr.to(device)
                sfh_output, sim_type_output, mass_sfr_output = AE(inputs.unsqueeze(1))
                sfh_output = sfh_output.squeeze(1)
                loss = compute_loss(sfh_output, inputs, sim_type_output, sim_labels, mass_sfr_output, mass_sfr, 1, 1)
                loss = loss[0]
                loss.backward()
                optimizer.step()
                scheduler.step() 
                
                total_loss += loss.item()

            total_loss /= len(train_loader)
            losses.append(total_loss)
            
            print(f'Epoch [{epoch}/{num_epochs}], Training Loss: {total_loss:.4f}')

        print('Training complete')
        return losses