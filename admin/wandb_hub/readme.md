## July 31 Sync with Chris re: W&B Jupyter Hub

- How many CPUs and how much RAM per container?
    - shows 8 cores, 32 GB
    - limit pod to 8GB, 2CPUs, 2GPUs
- Is it possible to do 2 GPUs per container?
    - It is possible. They're running 2 GPUs per docker right now.
- Persistent space
    - Chris will turn it on
    - 10GB
- What should be mounted
    - next docker build will start in home directory
    - can set env variable to clone a repo other than ml-class
        - [ ] send chris repo to clone
- Github access
    - should store username and password
- Admin (see other sessions, etc)?
    - done
- Troubleshooting (how to handle frozen sessions, for example)?
    - right-click in terminal, click Refresh Terminal
    - if people get "invalid code" messages, have them sign up for wandb again
    - [ ] send chris the schedule to pre-launch the cluster
- Is it possible to run Docker inside of container? (can the container run privileged?)

https://hub.wandb.us/hub/login
https://hub.wandb.us/hub/login?gpu=true

## Things that should be set in environment

```sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=.
alias ll="ls -lh"
```
