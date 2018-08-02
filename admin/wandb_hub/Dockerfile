FROM ufoym/deepo:latest

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    apt-get update && \
    $APT_INSTALL curl && \
    curl -sL https://deb.nodesource.com/setup_8.x | bash && \
    $APT_INSTALL nodejs sudo htop ssh redis-server parallel && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

ENV NB_USER=jovyan \
    HOME=/home/jovyan \
    SHELL=/bin/bash \
    NB_UID=1000 \
    NB_GID=100 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    LANGUAGE=C.UTF-8 \
    LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:/usr/local/nvidia/lib64

WORKDIR $HOME

ADD fix-permissions /usr/local/bin/fix-permissions

# Create wandb user with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
# TODO: This is awful, brittle, and puts shit in /home
RUN groupadd wheel -g 11 && \
    echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    usermod -aG staff,root $NB_USER && \
    chmod g+w /etc/passwd && \
    fix-permissions /usr/local/bin && \
    fix-permissions /usr/local/lib/python3.6/dist-packages/__pycache__ && \
    fix-permissions /usr/local/share && \
    fix-permissions /usr/local/etc && \
    mkdir /usr/etc && \
    fix-permissions /usr/etc && \
    fix-permissions /home && \
    echo "$NB_USER ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/notebook

USER $NB_UID

RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    $PIP_INSTALL git+https://github.com/data-8/nbgitpuller \
    jupyterhub==0.9.1 \
    jupyterlab==0.32.1 \
    pipenv \
    git+https://github.com/wandb/client && \
    rm -rf /tmp/*

USER root

COPY jupyterlab-wandb /usr/local/share/jupyterlab-wandb

RUN jupyter serverextension enable --py nbgitpuller --sys-prefix && \
    jupyter labextension install /usr/local/share/jupyterlab-wandb --no-build && \
    jupyter labextension install @jupyterlab/hub-extension@0.9.2 && \
    npm install -g serverless && \
    rm -rf $HOME/.cache && \
    rm -rf $HOME/.npm

COPY start.sh /usr/local/bin/
COPY wandb-git.sh /usr/local/bin/
RUN fix-permissions /usr/local/bin/start.sh && \
    fix-permissions /usr/local/bin/wandb-git.sh
USER $NB_UID