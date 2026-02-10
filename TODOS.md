# TODO 
- Chores
  - Move dataset transforms into their own module
  - Instead of get_kept_links, load links from the config (also adapt the dataset)
  - create src folder

- Run different configs
  - Different models
  - Different configs for LSTM architecture
    - More layers after LSTM part
    - Bigger hidden layers for LSTM part
  - training without socks in data
  
  - train old data with amplifying feet
    - train with very high feet and knee loss 

  - Split using folders
    - upload training / validation / test zip

- compare models with running in circles data 2025-12-02_12-24-03-all-felix
- Check load transformations for inference (does it load random rotation????)
- Check and compare metrics exhaustive -> possibly amplify feet in loss
  - The arms and feet find out what 80% accuracy means
  

# Done

- Move dataset code into a separate module
- Move models into their own modules
- Move link loss into its own module
- Convert config into a dataclass
- Split train and test methods
- Extend metrics (replace accuracy with L1 distance; report link loss and MSE separately)
- During training, save the loss in the model folder
- report link loss and MSE separately

Configs
  - Try "remove noise signals on the floor" with lstm
  - Rotate data again
  - Try roi_size 4 again
  - exhaustive offset strategy with lstm
  - active_field_min_value (e.g. 142, 145, 138)
  - training without socks
  - Try "remove noise signals on the floor"
  - increase patience to 5
  - only legs and hip with lstm
  - LSTM roi_size 4 + exhaustive + remove noise
  - Rotate data again
  - Increase History len to 50
