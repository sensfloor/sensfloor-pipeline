- Move dataset transforms into their own module
- Instead of get_kept_links, load links from the config (also adapt the dataset)

- Create actual walking sequences for (LSTM) dataset

- Run different configs
  - Different models
  - Different configs for LSTM architecture
  - training without socks in data
 
  - only legs and hip with lstm
  - LSTM roi_size 4 + exhaustive + remove noise
  - Rotate data again
  - Increase History len to 50

- Analyse field activity for each run
- Check load transformations for inference (does it load random rotation????)
- Check and compare exhaustive metrics -> possibly amplify feet in loss