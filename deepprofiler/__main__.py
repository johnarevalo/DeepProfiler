import tensorflow as tf
import json
import os
import copy
import click

import comet_ml

import deepprofiler.dataset.compression
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.indexing
import deepprofiler.dataset.illumination_statistics
import deepprofiler.dataset.metadata
import deepprofiler.dataset.utils
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.sampling
import deepprofiler.learning.training
import deepprofiler.learning.tf2train
import deepprofiler.learning.profiling
import deepprofiler.download.normalize_bbbc021_metadata


# Main interaction point
@click.group()
@click.option("--root", prompt="Root directory for DeepProfiler experiment",
              help="Root directory for DeepProfiler experiment",
              type=click.Path("r"))
@click.option("--config", default=None,
              help="Path to existing config file",
              type=click.STRING)
@click.option("--cores", default=0,
              help="Number of CPU cores for parallel processing (all=0)",
              type=click.INT)
@click.option("--gpu", default="0",
              help="GPU device id",
              type=click.STRING)
@click.option("--exp", default="results",
              help="Name of experiment",
              type=click.STRING)
@click.option("--single-cells", default="single-cells",
              help="Name of single cell export folder",
              type=click.STRING)
@click.option("--metadata", default='index.csv',
              help="Name of metadata file, default index.csv",
              type=click.STRING)
@click.option("--logging", default=None,
              help="Path to file with comet.ml API key",
              type=click.STRING)


@click.pass_context
def cli(context, root, config, exp, cores, gpu, single_cells, metadata, logging):
    dirs = {
        "root": root,
        "locations": root + "/inputs/locations/",  # TODO: use os.path.join()
        "config": root + "/inputs/config/",
        "images": root + "/inputs/images/",
        "metadata": root + "/inputs/metadata/",
        "intensities": root + "/outputs/intensities/",
        "compressed_images": root + "/outputs/compressed/images/",
        "single_cell_set": root + "/outputs/" + single_cells + "/",
        "results": root + "/outputs/" + exp + "/",
        "checkpoints": root + "/outputs/" + exp + "/checkpoint/",
        "logs": root + "/outputs/" + exp + "/logs/",
        "summaries": root + "/outputs/" + exp + "/summaries/",
        "features": root + "/outputs/" + exp + "/features/"
    }

    config = dirs["config"] + "/" + config
    context.obj["cores"] = cores
    context.obj["gpu"] = gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # Load configuration file
    if config is not None and os.path.isfile(config):
        with open(config, "r") as f:
            params = json.load(f)

        # Override paths defined by user
        if "paths" in params.keys():
            for key, value in dirs.items():
                if key not in params["paths"].keys():
                    params["paths"][key] = dirs[key]
                else:
                    dirs[key] = params["paths"][key]
        else:
            params["paths"] = copy.deepcopy(dirs)

        if os.path.isdir(dirs["root"]):
            for k in ["results", "checkpoints", "logs", "summaries", "features"]:
                os.makedirs(dirs[k], exist_ok=True)

        # Update references
        params["experiment_name"] = exp
        params["paths"]["index"] = params["paths"]["metadata"] + metadata
        if metadata != 'index.csv':
            params["paths"]["sc_index"] = os.path.join(params["paths"]["single_cell_set"], metadata)
        else:
            params["paths"]["sc_index"] = os.path.join(params["paths"]["single_cell_set"], 'sc-metadata.csv')
        context.obj["config"] = params
        if logging:
            with open(os.path.join(dirs["config"], logging), "r") as f:
                logging_params = json.load(f)
                if logging_params["log_type"] == "comet_ml":
                    context.obj["config"]["train"]["comet_ml"] = {}
                    context.obj["config"]["train"]["comet_ml"]["api_key"] = logging_params["api_key"]
                    context.obj["config"]["train"]["comet_ml"]["project_name"] = logging_params["project_name"]
    elif context.invoked_subcommand != 'setup':
        raise Exception("Config does not exists; make sure that the file exists in /inputs/config/")

    context.obj["dirs"] = dirs


# Optional tool: Create the support file and folder structure in a root directory
@cli.command()
@click.pass_context
def setup(context):
    for path in context.obj["dirs"].values():
        if not os.path.isdir(path):
            print("Creating directory: ", path)
            os.makedirs(path)
        else:
            print("Directory exists: ", path)
    context.obj["config"] = {}
    context.obj["config"]["paths"] = context.obj["dirs"]


# Optional tool: Download and prepare the BBBC021 dataset
@cli.command()
@click.pass_context
def download_bbbc021(context):
    context.invoke(setup)
    deepprofiler.download.normalize_bbbc021_metadata.normalize_bbbc021_metadata(context)
    print("BBBC021 download and preparation complete!")


# First tool: Compute illumination statistics and compress images
@cli.command()
@click.pass_context
def prepare(context):
    metadata = deepprofiler.dataset.metadata.read_plates(context.obj["config"]["paths"]["index"])
    process = deepprofiler.dataset.utils.Parallel(context.obj["config"], numProcs=context.obj["cores"])
    process.compute(deepprofiler.dataset.illumination_statistics.calculate_statistics, metadata)
    print("Illumination complete!")
    metadata = deepprofiler.dataset.metadata.read_plates(context.obj["config"]["paths"]["index"])  # reinitialize generator
    process.compute(deepprofiler.dataset.compression.compress_plate, metadata)
    print("Compression complete!")


# Second tool: Export single cells for training
@cli.command()
@click.pass_context
def export_sc(context):
    if context.parent.obj["config"]["prepare"]["compression"]["implement"]:
        context.parent.obj["config"]["paths"]["images"] = context.obj["config"]["paths"]["compressed_images"]
    dset = deepprofiler.dataset.image_dataset.read_dataset(context.obj["config"])
    deepprofiler.dataset.sampling.export_dataset(context.obj["config"], dset)
    print("Single-cell sampling complete.")


# Third tool: Train a network
@cli.command()
@click.option("--epoch", default=1)
@click.option("--seed", default=None)
@click.pass_context
def train(context, epoch, seed):
    if context.parent.obj["config"]["prepare"]["compression"]["implement"]:
        context.parent.obj["config"]["paths"]["images"] = context.obj["config"]["paths"]["compressed_images"]

    if context.parent.obj["config"]["train"]["model"]["crop_generator"] == 'crop_generator':
        dset = deepprofiler.dataset.image_dataset.read_dataset(context.obj["config"], mode='train')
        deepprofiler.learning.training.learn_model(context.obj["config"], dset, epoch, seed)
    else:
        deepprofiler.learning.training.learn_model(context.obj["config"], None, epoch, seed)


# Third tool (b): Train a network with TF dataset
@cli.command()
@click.option("--epoch", default=1)
@click.pass_context
def traintf2(context, epoch):
    deepprofiler.learning.training.learn_model_v2(context.obj["config"], epoch)


# Fourth tool: Profile cells and extract features
@cli.command()
@click.pass_context
@click.option("--part",
              help="Part of index to process",
              default=-1, 
              type=click.INT)
def profile(context, part):
    if context.parent.obj["config"]["prepare"]["compression"]["implement"]:
        context.parent.obj["config"]["paths"]["images"] = context.obj["config"]["paths"]["compressed_images"]
    config = context.obj["config"]
    if part >= 0:
        partfile = "index-{0:03d}.csv".format(part)
        config["paths"]["index"] = context.obj["config"]["paths"]["index"].replace("index.csv", partfile)
    dset = deepprofiler.dataset.image_dataset.read_dataset(context.obj["config"], mode='profile')
    deepprofiler.learning.profiling.profile(context.obj["config"], dset)
    

# Auxiliary tool: Split index in multiple parts
@cli.command()
@click.pass_context
@click.option("--parts", 
              help="Number of parts to split the index",
              type=click.INT)
def split(context, parts):
    if context.parent.obj["config"]["prepare"]["compression"]["implement"]:
        context.parent.obj["config"]["paths"]["images"] = context.obj["config"]["paths"]["compressed_images"]
    deepprofiler.dataset.indexing.split_index(context.obj["config"], parts)


if __name__ == "__main__":
    cli(obj={})
