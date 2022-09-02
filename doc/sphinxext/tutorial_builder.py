from pathlib import Path

from jinja2 import Environment
import yaml


TEMPLATE = """
:notoc:

.. _tutorial:

User guide and tutorial
=======================
{% for section in sections %}
{{ section.header }}
{% for page in section.pages %}
.. grid:: 1
  :gutter: 2

  .. grid-item-card::

    .. grid:: 2

      .. grid-item::
        :columns: 3

        .. image:: ./tutorial/{{ page }}.svg
          :target: ./tutorial/{{ page }}.html

      .. grid-item::
        :columns: 9
        :margin: auto

        .. toctree::
          :maxdepth: 2

          tutorial/{{ page }}
{% endfor %}
{% endfor %}
"""


def main(app):

    content_yaml = Path(app.builder.srcdir) / "tutorial.yaml"
    tutorial_rst = Path(app.builder.srcdir) / "tutorial.rst"

    tutorial_dir = Path(app.builder.srcdir) / "tutorial"
    tutorial_dir.mkdir(exist_ok=True)

    with open(content_yaml) as fid:
        sections = yaml.load(fid, yaml.BaseLoader)

    for section in sections:
        title = section["title"]
        section["header"] = "\n".join([title, "-" * len(title)]) if title else ""

    env = Environment().from_string(TEMPLATE)
    content = env.render(sections=sections)

    with open(tutorial_rst, "w") as fid:
        fid.write(content)


def setup(app):
    app.connect("builder-inited", main)
