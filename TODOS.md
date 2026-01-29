- Move dataset transforms into their own module
- Instead of get_kept_links, load links from the config (also adapt the dataset)

- Create actual walking sequences for (LSTM) dataset

- Run different configs
  - Different models
  - Different configs for LSTM architecture
  - training without socks in data
  - Try "remove noise signals on the floor" with lstm
  - Rotate data again
  - Try roi_size 4 again
  - exhaustive offset strategy with lstm
  - only legs and hip with lstm

- Analyse field activity for each run
- Check load transformations for inference (does it load random rotation????)
- Check and compare exhaustive metrics -> possibly amplify feet in loss