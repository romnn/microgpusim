### PTX reference bson parser

This is the bison and flex based PTX parser of AccelSim.

It has been extracted from AccelSim to allow for quick comparisons.

##### Build
**Note**: Please make sure you have a recent version of bison and flex installed.

```bash
cargo build -p ptxbison

# you can also specify a path to another bison version
BISON_PATH=/usr/local/Cellar/bison/3.8.2/bin/bison cargo build -p ptxbison
```

##### Usage

```bash
# todo
```
