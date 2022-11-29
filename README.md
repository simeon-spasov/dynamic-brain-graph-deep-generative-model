# Instructions

- Install pipenv in home directory
  - pip install --user pipenv
- Add pipenv to $PATH. See WARNING message after pipenv installation.
  - export PATH="$HOME/.local/bin:$PATH"
- Setup repo
  - Clone repo:
    - git clone https://github.com/ajrcampbell/dynamic-graph-generative-model.git
  - Install dependencies from Pipfile
    - cd dynamic-graph-generative-model
    - git checkout dev
    - pipenv install
- Activate pipenv shell and run main.py training script:
  - pipenv shell
  - python main.py

- Updates Pipfile.lock and extracts a requirements.txt file
  - pipenv lock -r > requirements.txt

- Paper writeup
  - https://www.overleaf.com/project/62b05e8ad70a315088a42a3d

- Data
  - https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP1200-DenseConnectome+PTN+Appendix-July2017.pdf

# Cannot find cuDNN on Enterprise

"OSError: libcudnn.so.8: cannot open shared object file: No such file or directory"

- Enter virtual env
  - pipenv run

- Uninstall pytorch
  - pipenv uninstall torch

- Reinstall pytroch via pip via pipenv
  - pipenv run pip install torch