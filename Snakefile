n_events = 500000
switch_everys = [
    # 60,
    120, 300, 600, 900, 1200, 1800, 3600
]

targets = expand(
    "output/switch={switch_every}/results.csv",
    switch_every=switch_everys
)

rule all:
    input:
        ["output/ate.csv"] + targets


rule run:
    output:
        results="output/switch={switch_every}/results.csv",
        cfg="output/switch={switch_every}/config.csv",
    shell:
        """
        python rideshare-incremental.py with \
        switch_every={wildcards.switch_every} \
        output={output.results:q} \
        config_output={output.cfg:q} \
        n_events={n_events} \
        k=999\
        batch_size=1000 \
        seed=42
        """


rule ate:
    output:
        "output/ate.csv"
    shell:
        """
        python compute-ate.py with \
        n_events={n_events} \
        output={output} \
        k=10000 \
        batch_size=1000
        """
