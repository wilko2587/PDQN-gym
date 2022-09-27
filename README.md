## PQDN on gym platform
### James Wilkinson

We present a PDQN model applied to the platform domain.
Please install both requirements.txt, and follow the instructions to setup gym-platform below.

The PDQN agent, pytorch infrastructure, and training/playing proceedure is built in PDQN.py.

run_PDQN.py should run out-of-the-box, and begin training the agent.

## Dependencies (for gym-platform)

- Python 3.5+ (tested with 3.5 and 3.6)
- gym 0.10.5
- pygame 1.9.4
- numpy

## Setup for pytorch
Please ensure pytorch is properly installed on the device.

## Installing gym-platform (for gym-platform)

Install this as any other OpenAI gym environment:

    git clone https://github.com/cycraig/gym-platform
    cd gym-platform
    pip install -e '.[gym-platform]'
    
or 

    pip install -e git+https://github.com/cycraig/gym-platform#egg=gym_platform

### gym-platform citation
    @inproceedings{Masson2016ParamActions,
        author = {Masson, Warwick and Ranchod, Pravesh and Konidaris, George},
        title = {Reinforcement Learning with Parameterized Actions},
        booktitle = {Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence},
        year = {2016},
        location = {Phoenix, Arizona},
        pages = {1934--1940},
        numpages = {7},
        publisher = {AAAI Press},
    }
