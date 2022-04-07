# sdow

natural language understanding for six degrees of wikipedia

## usage

install requirements.

```sh
poetry install
```

auth and choose start and target pages.

```sh
export COHERE_API_TOKEN='generated from https://os.cohere.ai/'
export SDOW_START='https://en.wikipedia.org/wiki/Monty_Hall_problem'
export SDOW_TARGET='https://en.wikipedia.org/wiki/Event_(probability_theory)'
```

exercise the classic solution.

```sh
poetry run sdow --classic
```

exercise the NLU solution.

```sh
poetry run sdow
```

![](./demo.gif)
