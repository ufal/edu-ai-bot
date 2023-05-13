import argparse
import os
import glob
import xml.etree.ElementTree as ET
import yaml
from yaml.loader import SafeLoader

from edubot.remote_services import RemoteServiceHandler


def process_file(inp_file, remote_service_handler, ofd, sep="\t", cached=None):
    print(inp_file)
    # extract all the contents from the <category> tags using xml parser
    # extract pairs from the <pattern> and <template> tags
    # save the contents to a list
    parser = ET.XMLParser(encoding="utf-8")
    tree = ET.parse(inp_file, parser=parser)
    root = tree.getroot()
    for child in root:
        if child.tag == "category":
            pattern = child.find("pattern").text
            if pattern is not None:
                pattern = pattern.strip().lower()
            else:
            template = child.find("template").text
            if template is not None:
                templates = [template.strip().lower()]
            else:
                templates = []
                for inner in child.find("template"):
                    if inner.text is not None:
                        templates.append(inner.text.strip().lower())
                    for item in inner:
                        txt = item.text
                        if txt is not None:
                            templates.append(txt.strip().lower())
            if cached is not None and pattern in cached:
                continue
            if len(templates) == 0 or pattern is None:
                continue
            for tmpl in templates:
                try:
                    print(sep.join([pattern.lower().strip(),
                                    tmpl.lower().strip(),
                                    remote_service_handler.translate_en2cs(pattern.lower().strip()).strip(),
                                    remote_service_handler.translate_en2cs(tmpl.lower().strip())]).strip(),
                          file=ofd, flush=True)
                except Exception as e:
                    print(e)
                    continue


def main(args):
    processed = set()
    with open(args.processed, 'rt') as fd:
        for line in fd:
            line = line.split("\t")
            processed.add(line[0])

    if os.path.isdir(args.input):
        files = glob.glob(os.path.join(args.input, "*.aiml"))
    else:
        files = [args.input]
    with open(args.config, 'rt') as fd:
        config = yaml.load(fd, Loader=SafeLoader)
    remote_service_handler = RemoteServiceHandler(config)
    with open(args.output, 'at') as ofd:
        for inp_file in files:
            process_file(inp_file, remote_service_handler, ofd, sep=args.separator, cached=processed)
            print("Done with file: ", inp_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to input file/directory")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--processed", type=str, help="Path to output file")
    parser.add_argument("--output", type=str, help="Path to output file")
    parser.add_argument("--separator", type=str, default="\t")
    args = parser.parse_args()
    main(args)
