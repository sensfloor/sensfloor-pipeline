# TODO 
- Chores
  - Move dataset transforms into their own module
  - Instead of get_kept_links, load links from the config (also adapt the dataset)

- Run different configs
  - Different models
  - Different configs for LSTM architecture
  - training without socks in data

  - Split using folders

 - ideas
   - amplify feet in loss

- compare models with running in circles data 2025-12-02_12-24-03-all-felix
- Check load transformations for inference (does it load random rotation????)
- Check and compare exhaustive metrics -> possibly amplify feet in loss

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
