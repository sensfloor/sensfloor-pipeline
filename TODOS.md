- Move dataset transforms into their own module
- Instead of get_kept_links, load links from the config (also adapt the dataset)

- Create actual walking sequences for (LSTM) dataset
- Run different configs
  - active_field_min_value (e.g. 142, 145, 138)
  - Different models
  - Different configs for LSTM architecture
