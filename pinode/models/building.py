import os
import time
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import pad_sequence

from util.util import format_elapsed_time

SAVE_NAME = os.path.join('data', "building_parameters_")

class BuildingModule(nn.Module):
    
    def __init__(self, h=60):
        super().__init__()
        
        self.h = h
        
        self.lambda_12 = nn.Parameter(torch.randn(1,1), True)
        self.lambda_23 = nn.Parameter(torch.randn(1,1), True)
        self.lambda_e1 = nn.Parameter(torch.randn(1,1), True)
        self.lambda_e2 = nn.Parameter(torch.randn(1,1), True)
        self.lambda_e3 = nn.Parameter(torch.randn(1,1), True)
        self.lambda_s1 = nn.Parameter(torch.randn(1,1), True)
        self.lambda_s2 = nn.Parameter(torch.randn(1,1), True)
        self.lambda_s3 = nn.Parameter(torch.randn(1,1), True)
        self.lambda_h1 = nn.Parameter(torch.randn(1,1), True)
        self.lambda_h2 = nn.Parameter(torch.randn(1,1), True)
        self.lambda_h3 = nn.Parameter(torch.randn(1,1), True)
        self.lambda_c1 = nn.Parameter(torch.randn(1,1), True)
        self.lambda_c2 = nn.Parameter(torch.randn(1,1), True)
        self.lambda_c3 = nn.Parameter(torch.randn(1,1), True)
        
        #self.A
        #self.act = nn.Tanh()
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("\nGPU acceleration on!")
        else:
            self.device = "cpu"
        
        self.c = nn.Parameter(torch.FloatTensor([10665991.0, 9000000*3, 7953253.0]), True).reshape(1,1,-1)
        
        
    def forward(self, x0, u_):
        """
        x: temperatures of Room 1, 2, 3
        u: tensor with the external temperature, the solar irradiation, then 
            the power input of Room 1, 2, 3
        """
        
        x = x0.clone()
        u = u_.clone()
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)        
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        if len(u.shape) == 1:
            u = u.unsqueeze(0)       
        if len(u.shape) == 2:
            u = u.unsqueeze(1)
                        
        predictions = torch.empty((u.shape[0], u.shape[1], x.shape[2]))
        predictions[:,0,:] = x.squeeze(1).clone()
        
        for t in range(u.shape[1]-1):
            
            A = torch.zeros(u.shape[0], 3, 3).to(self.device)
            A[:,0,1] = torch.exp(self.lambda_12) * (1/x[:,0,0] - 1/x[:,0,1])
            A[:,1,0] = -torch.exp(self.lambda_12) * (1/x[:,0,0] - 1/x[:,0,1])
            A[:,1,2] = torch.exp(self.lambda_23) * (1/x[:,0,1] - 1/x[:,0,2])
            A[:,2,1] = -torch.exp(self.lambda_23) * (1/x[:,0,1] - 1/x[:,0,2])
            
            B = torch.zeros(u.shape[0], 3, u.shape[2]).to(self.device)
            B[:,0,0] = torch.exp(self.lambda_e1) * (1/x[:,0,0] - 1/u[:,t,0])
            B[:,1,0] = torch.exp(self.lambda_e2) * (1/x[:,0,1] - 1/u[:,t,0])
            B[:,2,0] = torch.exp(self.lambda_e3) * (1/x[:,0,2] - 1/u[:,t,0])
            B[:,0,1] = torch.exp(self.lambda_s1)
            B[:,1,1] = torch.exp(self.lambda_s2)
            B[:,2,1] = torch.exp(self.lambda_s3)
            B[:,0,2] = torch.exp(self.lambda_h1)
            B[:,1,3] = torch.exp(self.lambda_h2)
            B[:,2,4] = torch.exp(self.lambda_h3)
            B[:,0,5] = torch.exp(self.lambda_c1)
            B[:,1,6] = torch.exp(self.lambda_c2)
            B[:,2,7] = torch.exp(self.lambda_c3)
                        
            x = x * torch.exp(self.h * (torch.bmm(A.clone(), x.mT) 
                                        + torch.bmm(B.clone(), u[:,t,:].unsqueeze(-1))).mT / self.c)
            predictions[:,t+1,:] = x.squeeze(1).clone()
                        
            # Trick needed since some sequences are padded
            predictions[torch.where(u[:, t+1, 0] < 1e-6)[0], t+1, :] = -1
            
        return predictions

class Building:
    def __init__(self, data, h, lr=0.001, verbose=4, try_to_load=True, name='default'):
        self.data = data
        self.X = data.copy().values
        self.columns = data.columns
        self.x_col = [2, 3, 4] 
        self.u_col = [1, 0, 5, 6, 7, 9, 10, 11] 
        self.interval = 15
        
        self.heating = True
        self.cooling = True
        
        self.times = []
        self.train_losses = []
        self.validation_losses = []
        
        self.batch_size = 64
        self.verbose = verbose
        self.name = name
        
        self.model = BuildingModule(h*self.interval)
        
        sequences = torch.load(os.path.join('data', 'sequences.pt'))
        self.heating_sequences = sequences["heating_sequences"]
        self.cooling_sequences = sequences["cooling_sequences"]

        self.train_test_validation_separation()
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("\nGPU acceleration on!")
        else:
            self.device = "cpu"
            
        self.model = self.model.to(self.device)
        
        self.loss = F.mse_loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        if try_to_load:
            self.load()
    
    def train_test_validation_separation(self, validation_percentage: float = 0.2, test_percentage: float = 0.0) -> None:
        """
        Function to separate the data into training and testing parts. The trick here is that
        the data is not actually split - this function actually defines the sequences of
        data points that are in the training/testing part.

        Args:
            validation_percentage:  Percentage of the data to keep out of the training process
                                    for validation
            test_percentage:        Percentage of data to keep out of the training process for
                                    the testing

        Returns:
            Nothing, in place definition of all the indices
        """

        # Sanity checks: the given inputs are given as percentage between 0 and 1
        if 1 <= validation_percentage <= 100:
            validation_percentage /= 100
            print("The train-test-validation separation rescaled the validation_percentage between 0 and 1")
        if 1 <= test_percentage <= 100:
            test_percentage /= 100
            print("The train-test-validation separation rescaled the test_percentage between 0 and 1")

        # Prepare the lists
        self.train_sequences = []
        self.validation_sequences = []
        self.test_sequences = []

        for sequences in [self.heating_sequences, self.cooling_sequences]:
            if len(sequences) > 0:
                # Given the total number of sequences, define aproximate separations between training
                # validation and testing sets
                train_validation_sep = int((1 - test_percentage - validation_percentage) * len(sequences))
                validation_test_sep = int((1 - test_percentage) * len(sequences))

                # Little trick to ensure training, validation and test sequences are completely distinct
                while True:
                    if (sequences[train_validation_sep - 1][1] < sequences[train_validation_sep][0]) | (train_validation_sep == 1):
                        break
                    train_validation_sep -= 1
                if test_percentage > 0.:
                    while True:
                        if (sequences[validation_test_sep - 1][1] < sequences[validation_test_sep][0]) | (validation_test_sep == 1):
                            break
                        validation_test_sep -= 1

                # Prepare the lists
                self.train_sequences += sequences[:train_validation_sep]
                self.validation_sequences += sequences[train_validation_sep:validation_test_sep]
                self.test_sequences += sequences[validation_test_sep:]
    
    def batch_iterator(self, iterator_type: str = "train", batch_size: int = None, shuffle: bool = True) -> None:
        """
        Function to create batches of the data with the wanted size, either for training,
        validation, or testing

        Args:
            iterator_type:  To know if this should handle training, validation or testing data
            batch_size:     Size of the batches
            shuffle:        Flag to shuffle the data before the batches creation

        Returns:
            Nothing, yields the batches to be then used in an iterator
        """

        # Firstly control that the training sequences exist - create them otherwise
        if self.train_sequences is None:
            self.train_test_validation_separation()
            print("The Data was not separated in train, validation and test --> the default 70%-20%-10% was used")

        # If no batch size is given, define it as the default one
        if batch_size is None:
            batch_size = self.batch_size

        # Copy the indices of the correct type (without first letter in case of caps)
        if "rain" in iterator_type:
            sequences = self.train_sequences
        elif "alidation" in iterator_type:
            sequences = self.validation_sequences
        elif "est" in iterator_type:
            sequences = self.test_sequences
        else:
            raise ValueError(f"Unknown type of batch creation {iterator_type}")

        # Shuffle them if wanted
        if shuffle:
            np.random.shuffle(sequences)

        # Define the right number of batches according to the wanted batch_size - taking care of the
        # special case where the indicies ae exactly divisible by the batch size, which can induce
        # an additional empty batch breaking the simulation down the line
        n_batches = int(np.ceil(len(sequences) / batch_size))

        # Iterate to yield the right batches with the wanted size
        for batch in range(n_batches):
            yield sequences[batch * batch_size: (batch + 1) * batch_size]
    
    def build_input_output_data(self, sequences):
        
        if isinstance(sequences, tuple):
            sequences = [sequences]
            
        # Arx has a warm start
        sequences = [(seq[0] + 12, seq[1]) for seq in sequences]
        
        y = [torch.FloatTensor(self.X[seq[0]:seq[1], self.x_col].copy()) for seq in sequences]
        x = [torch.FloatTensor(self.X[seq[0]:seq[1], :].copy()) for seq in sequences]
                
        # Build the final results by taking care of the batch_size=1 case
        if len(sequences) > 1:
            batch_x = pad_sequence(x, batch_first=True, padding_value=-1)
            batch_y = pad_sequence(y, batch_first=True, padding_value=-1)
        else:
            batch_x = x[0].view(1, x[0].shape[0], -1)
            batch_y = y[0].view(1, y[0].shape[0], -1)

        # Return everything
        return batch_x.to(self.device), batch_y.to(self.device)
        
    def simulate(self, sequences):
                 
        x, y = self.build_input_output_data(sequences)
        predictions = self.model(x[:, 0, self.x_col], x[:, :, self.u_col])
                    
        return predictions, y
    
    def fit(self, n_epochs: int = None, show_plot: bool = True, print_each=5) -> None:
        """
        General function fitting a model for several epochs, training and evaluating it on the data.

        Args:
            n_epochs:         Number of epochs to fit the model, if None this takes the default number
                                defined by the parameters
            show_plot:        Flag to set to False if you don't want to have the plot of losses shown

        Returns:
            Nothing
        """

        self.times.append(time.time())

        if self.verbose > 0:
            print("\nTraining starts!\n")

        # If no special number of epochs is given, take the default one
        if n_epochs is None:
            n_epochs = self.n_epochs

        # Define the best loss, taking the best existing one or a very high loss
        best_loss = np.min(self.validation_losses) if len(self.validation_losses) > 0 else np.inf

        # Assess the number of epochs the model was already trained on to get nice prints
        trained_epochs = len(self.train_losses)
        
        if self.verbose> 0:
            print('\t\tTraining loss\t\tValidation loss')

        for epoch in range(trained_epochs, trained_epochs + n_epochs):

            # Start the training, define a list to retain the training losses along the way
            self.model.train()
            train_losses = []


            # Create training batches and run through them, using the batch_iterator function, which has to be defined
            # independently for each subclass, as different types of data are handled differently
            for num_batch, batch_sequences in enumerate(self.batch_iterator(iterator_type="train")):

                self.optimizer.zero_grad()
                
                # Compute the loss of the batch and store it
                predictions, truth = self.simulate(batch_sequences)
                loss = self.loss(predictions/10, truth/10)

                # Compute the gradients and take one step using the optimizer
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(float(loss))

            # Compute the average loss of the training epoch and print it
            train_loss = sum(train_losses) / len(train_losses)
            self.train_losses.append(train_loss)
            
            validation_losses = []
            _validation_losses = []

            # Create validation batches and run through them. Note that we use larger batches
            # to accelerate it a bit, and there is no need to shuffle the indices
            for num_batch, batch_sequences in enumerate(self.batch_iterator(iterator_type="validation", batch_size=2 * self.batch_size, shuffle=False)):

                # Compute the loss, in the torch.no_grad setting: we don't need the model to
                # compute and use gradients here, we are not training
                self.model.eval()
                with torch.no_grad():
                    predictions, truth = self.simulate(batch_sequences)
                    loss = self.loss(predictions/10, truth/10)
                    validation_losses.append(float(loss))

            # Compute the average validation loss of the epoch and print it
            validation_loss = sum(validation_losses) / len(validation_losses)
            self.validation_losses.append(validation_loss)
            
            # Compute the average loss of the training epoch and print it
            if epoch == 0 or (epoch % print_each == print_each-1):
                print(f"Epoch {epoch + 1}:\t  {train_loss:.2E}\t\t   {validation_loss:.2E}")

            # Timing information
            self.times.append(time.time())

            if validation_loss < best_loss:
                self.save(name_to_add="best", verbose=1)
                best_loss = validation_loss

        if self.verbose > 0:
            best_epoch = np.argmin([x for x in self.validation_losses])
            print(f"\nThe best model was obtained at epoch {best_epoch + 1} wit loss {np.min([x for x in self.validation_losses]):.2E}")
            print(f"Total training time: {format_elapsed_time(self.times[0], self.times[-1])}")
            
        self.load()
            
    def save(self, name_to_add: str = None, verbose: int = 0):

        torch.save(
            {"model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "validation_losses": self.validation_losses,
                "times": self.times,
                },
            SAVE_NAME+self.name+'.pt',
        )

    def load(self):
        """
        Function trying to load an existing model, by default the best one if it exists. But for training purposes,
        it is possible to load the last state of the model instead.

        Args:
            load_last:  Flag to set to True if the last model checkpoint is wanted, instead of the best one

        Returns:
             Nothing, everything is done in place and stored in the parameters.
        """

        print("\nTrying to load a trained model...")
        try:
            # Build the full path to the model and check its existence

            assert os.path.exists(SAVE_NAME+self.name+'.pt'), f"The file {SAVE_NAME+self.name+'.pt'} doesn't exist."

            # Load the checkpoint
            checkpoint = torch.load(SAVE_NAME+self.name+'.pt', map_location=lambda storage, loc: storage)

            # Put it into the model
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if torch.cuda.is_available():
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
            self.train_losses = checkpoint["train_losses"]
            self.validation_losses = checkpoint["validation_losses"]
            self.times = checkpoint["times"]

            # Print the current status of the found model
            if self.verbose > 0:
                print(f"Found!\nThe model has been fitted for {len(self.train_losses)} epochs already, "
                      f"with loss {np.min(self.validation_losses): .5f}.")

        # Otherwise, keep an empty model, return nothing
        except AssertionError:
            print("No existing model was found!")