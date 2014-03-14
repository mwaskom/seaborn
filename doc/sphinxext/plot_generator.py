import sys
import os
import glob
import token
import tokenize
import shutil
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib import image
from matplotlib.figure import Figure


RST_TEMPLATE = """
.. _{sphinx_tag}:

{docstring}

.. image:: {img_file}

**Python source code:** :download:`[download source: {fname}]<{fname}>`

.. literalinclude:: {fname}
    :lines: {end_line}-
"""


INDEX_TEMPLATE = """

.. raw:: html

    <style type="text/css">
    .figure {{
        float: left;
        margin: 10px;
        width: 180px;
        height: 200px;
    }}

    .figure img {{
        display: inline;
        width: 170px;
        height: 170px;
        opacity:0.4;
        filter:alpha(opacity=40); /* For IE8 and earlier */
    }}

    .figure img:hover
    {{
        opacity:1.0;
        filter:alpha(opacity=100); /* For IE8 and earlier */
    }}

    .figure .caption {{
        width: 180px;
        text-align: center !important;
    }}
    </style>

.. _{sphinx_tag}:

Example gallery
===============

{toctree}

{contents}

.. raw:: html

    <div style="clear: both"></div>
"""


BANNER_JS_TEMPLATE = """

var banner_data = {banner_data};

banner_data.forEach(function(d, i) {{
  d.i = i;
}});

var height = 150,
    width = 900,
    imageHeight = 150,
    imageWidth = 150,
    zoomfactor = 0.1;

var banner = d3.select(".example-banner");

banner.style("height", height + "px")
      .style("width", width + "px")
      .style("margin-left", "auto")
      .style("margin-right", "auto");

var svg = banner.append("svg")
                .attr("width", width + "px")
                .attr("height", height + "px");

var anchor = svg.append("g")
                  .attr("class", "example-anchor")
                .selectAll("a")
                .data(banner_data.slice(0, 7));

anchor.exit().remove();

var anchor_elements = anchor.enter().append("a")
      .attr("xlink:href", function(d) {{ return d.url; }})
      .attr("xlink:title", function(d) {{ return d.title; }});

anchor_elements.append("svg:image")
      .attr("width", (1 - zoomfactor) * imageWidth)
      .attr("height", (1 - zoomfactor) * imageHeight)
      .attr("xlink:href", function(d){{ return d.thumb; }})
      .attr("xroot", function(d){{return d3.round(imageWidth * (d.i - 0.5));}})
      .attr("x", function(d){{return d3.round(imageWidth * (d.i - 0.5));}})
      .attr("y", d3.round(0.5 * zoomfactor * imageHeight))
      .attr("i", function(d){{return d.i;}})
     .on("mouseover", function() {{
              var img = d3.select(this);
              img.transition()
                    .attr("width", imageWidth)
                    .attr("height", height)
                    .attr("x", img.attr("xroot")
                               - d3.round(0.5 * zoomfactor * imageWidth))
                    .attr("y", 0);
              }})
     .on("mouseout", function() {{
              var img = d3.select(this);
              img.transition()
                    .attr("width", (1 - zoomfactor) * imageWidth)
                    .attr("height", (1 - zoomfactor) * height)
                    .attr("x", img.attr("xroot"))
                    .attr("y", d3.round(0.5 * zoomfactor * imageHeight));
              }});
"""


def create_thumbnail(infile, thumbfile,
                     width=300, height=300,
                     cx=0.5, cy=0.6, border=4):
    # this doesn't really matter, it will cancel in the end, but we
    # need it for the mpl API
    dpi = 100

    baseout, extout = os.path.splitext(thumbfile)
    im = image.imread(infile)
    rows, cols = im.shape[:2]
    x0 = int(cx * cols - 0.7 * width)
    y0 = int(cy * rows - 0.7 * height)
    thumb = im[y0: y0 + height,
               x0: x0 + width]
    thumb[:border, :, :3] = thumb[-border:, :, :3] = 0
    thumb[:, :border, :3] = thumb[:, -border:, :3] = 0

    extension = extout.lower()

    if extension == '.png':
        from matplotlib.backends.backend_agg \
            import FigureCanvasAgg as FigureCanvas
    elif extension == '.pdf':
        from matplotlib.backends.backend_pdf \
            import FigureCanvasPDF as FigureCanvas
    elif extension == '.svg':
        from matplotlib.backends.backend_svg \
            import FigureCanvasSVG as FigureCanvas
    else:
        raise ValueError("Can only handle extensions 'png', 'svg' or 'pdf'")

    fig = Figure(figsize=(float(width) / dpi, float(height) / dpi),
                 dpi=dpi)
    canvas = FigureCanvas(fig)

    ax = fig.add_axes([0, 0, 1, 1], aspect='auto',
                      frameon=False, xticks=[], yticks=[])

    ax.imshow(thumb, aspect='auto', resample=True,
              interpolation='bilinear')
    fig.savefig(thumbfile, dpi=dpi)
    return fig


def indent(s, N=4):
    """indent a string"""
    return s.replace('\n', '\n' + N * ' ')


class ExampleGenerator(object):
    """Tools for generating an example page from a file"""
    def __init__(self, filename, target_dir):
        self.filename = filename
        self.target_dir = target_dir
        self.extract_docstring()
        self.exec_file()

    @property
    def dirname(self):
        return os.path.split(self.filename)[0]

    @property
    def fname(self):
        return os.path.split(self.filename)[1]

    @property
    def modulename(self):
        return os.path.splitext(self.fname)[0]

    @property
    def pyfilename(self):
        return self.modulename + '.py'

    @property
    def rstfilename(self):
        return self.modulename + ".rst"

    @property
    def htmlfilename(self):
        return self.modulename + '.html'

    @property
    def pngfilename(self):
        pngfile =  self.modulename + '.png'
        return "_images/" + pngfile

    @property
    def thumbfilename(self):
        pngfile =  self.modulename + '_thumb.png'
        return "_images/" + pngfile

    @property
    def sphinxtag(self):
        return self.modulename

    @property
    def pagetitle(self):
        return self.docstring.strip().split('\n')[0].strip()

    def extract_docstring(self):
        """ Extract a module-level docstring
        """
        lines = open(self.filename).readlines()
        start_row = 0
        if lines[0].startswith('#!'):
            lines.pop(0)
            start_row = 1

        docstring = ''
        first_par = ''
        tokens = tokenize.generate_tokens(lines.__iter__().next)
        for tok_type, tok_content, _, (erow, _), _ in tokens:
            tok_type = token.tok_name[tok_type]
            if tok_type in ('NEWLINE', 'COMMENT', 'NL', 'INDENT', 'DEDENT'):
                continue
            elif tok_type == 'STRING':
                docstring = eval(tok_content)
                # If the docstring is formatted with several paragraphs,
                # extract the first one:
                paragraphs = '\n'.join(line.rstrip()
                                       for line in docstring.split('\n')
                                       ).split('\n\n')
                if len(paragraphs) > 0:
                    first_par = paragraphs[0]
            break

        self.docstring = docstring
        self.short_desc = first_par
        self.end_line = erow + 1 + start_row

    def exec_file(self):
        print("running {0}".format(self.filename))

        plt.close('all')
        my_globals = {'pl': plt,
                      'plt': plt}
        execfile(self.filename, my_globals)

        fig = plt.gcf()
        fig.canvas.draw()
        pngfile = os.path.join(self.target_dir,
                               self.pngfilename)
        self.html = "<img src=../%s>" % self.pngfilename
        fig.savefig(pngfile, dpi=75)
        create_thumbnail(pngfile, "examples/" + self.thumbfilename)

    def toctree_entry(self):
        return "   ./%s\n\n" % os.path.splitext(self.htmlfilename)[0]

    def contents_entry(self):
        return (".. figure:: ./{0}\n"
                "    :target: ./{1}\n"
                "    :align: center\n\n"
                "    :ref:`{2}`\n\n".format(self.thumbfilename,
                                            self.htmlfilename,
                                            self.sphinxtag))


def main(app):
    static_dir = os.path.join(app.builder.srcdir, '_static')
    target_dir = os.path.join(app.builder.srcdir, 'examples')
    source_dir = os.path.abspath(os.path.join(app.builder.srcdir,
                                              '..', 'examples'))
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    banner_data = []

    toctree = ("\n\n"
               ".. toctree::\n"
               "   :hidden:\n\n")
    contents = "\n\n"

    # Write individual example files
    for filename in glob.glob(os.path.join(source_dir, "*.py")):
        ex = ExampleGenerator(filename, target_dir)

        banner_data.append({"title": ex.pagetitle,
                            "url": os.path.join('examples', ex.htmlfilename),
                            "thumb": os.path.join(ex.thumbfilename)})
        shutil.copyfile(filename, os.path.join(target_dir, ex.pyfilename))
        output = RST_TEMPLATE.format(sphinx_tag=ex.sphinxtag,
                                     docstring=ex.docstring,
                                     end_line=ex.end_line,
                                     fname=ex.pyfilename,
                                     img_file=ex.pngfilename)
        with open(os.path.join(target_dir, ex.rstfilename), 'w') as f:
            f.write(output)

        toctree += ex.toctree_entry()
        contents += ex.contents_entry()

    if len(banner_data) < 10:
        banner_data = (4 * banner_data)[:10]

    # write index file
    index_file = os.path.join(target_dir, 'index.rst')
    with open(index_file, 'w') as index:
        index.write(INDEX_TEMPLATE.format(sphinx_tag="example-gallery",
                                          toctree=toctree,
                                          contents=contents))

    # write javascript include for front page
    js_file = os.path.join(static_dir, 'banner_data.js')
    with open(js_file, 'w') as js:
        js.write(BANNER_JS_TEMPLATE.format(
            banner_data=json.dumps(banner_data)))


def setup(app):
    app.connect('builder-inited', main)
