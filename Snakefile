switch_everys = [
    60, 120, 300, 600, 900, 1200, 1800, 3600
]

targets = expand(
    "output/switch={switch_every}/results.csv",
    switch_every=switch_everys
)

rule all:
    input:
        targets

rule run:
    output:
        results="output/switch={switch_every}/results.csv",
        cfg="output/switch={switch_every}/config.csv",
    shell:
        """
        python rideshare.py with \
        switch_every={wildcards.switch_every} \
        output={output.results:q} \
        config_output={output.cfg:q} \
        n_events=200000 \
        k=10000 \
        batch_size=10 \
        seed=42
        """


rule ate:
    output:
        "output/ate.json"
    shell:
        "python compute-ate.py with n_events=200000 output={output}"
