# llm-interpolate

[LLM](https://llm.datasette.io/) plugin for interpolating between embeddings

## Installation

Install this plugin in the same environment as LLM.
```bash
llm install llm-interpolate
```

## Usage

The plugin adds a new command, `llm interpolate`. This command takes the name of an 
[embedding collection](https://llm.datasette.io/en/stable/embeddings/cli.html#storing-embeddings-in-sqlite) 
and two ids from that collection that will act as start and end points.

First, populate a collection.
In this case we are embedding a song library using the [CLAP](https://github.com/vagos/llm-clap) model.
```bash
llm embed-multi -m clap songs --files mysongs "*.wav" 
```

Now we can interpolate between those song embeddings with five intermediate points:
```bash
llm interpolate songs "MyRapSong.wav" "MyContrySong.wav" -n 5
```
You can use the `-d` option to use a different embeddings database.

What this will do is try and find points in the embedding space in-between the given points.

The output should look something like this (truncated):
```json
[
    "MyRapSong.wav",
    "HipHopMeetsCountry.wav",
    "SmoothCountryRap.wav",
    "CountryVibes.wav",
    "MyCountrySong.wav"
]
```

This output can then be used to build a cohesive playlist using only a starting and ending song.

```bash
llm interpolate songs "MyRapSong.wav" "MyContrySong.wav" -n 5 | jq .[] > playlist.m3u
```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-interpolate
python3 -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
pip install -e '.[test]'
```
To run the tests:
```bash
pytest
```
