FROM gitlab.lrz.de:5005/tum-cps/commonroad-docker-submission/base:1.0.0

COPY ./planner /commonroad/planner

RUN git clone https://gitlab.lrz.de/tum-cps/commonroad-interactive-scenarios.git && \
    pip install -r ./commonroad-interactive-scenarios/requirements.txt

RUN git clone https://gitlab.lrz.de/tum-cps/commonroad-route-planner.git && \
    pip install  ./commonroad-route-planner

ENV PYTHONPATH="/commonroad-interactive-scenarios/:${PYTHONPATH}"

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY run.sh run.sh
RUN chmod a+x run.sh

ENTRYPOINT ./run.sh
