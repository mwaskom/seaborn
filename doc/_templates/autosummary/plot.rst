{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% block methods %}

Methods
~~~~~~~

.. rubric:: Specification methods

.. autosummary::
   :toctree: ./
   :nosignatures:

   ~Plot.add
   ~Plot.scale

.. rubric:: Subplot methods

.. autosummary::
   :toctree: ./
   :nosignatures:

   ~Plot.facet
   ~Plot.pair

.. rubric:: Customization methods

.. autosummary::
   :toctree: ./
   :nosignatures:

   ~Plot.layout
   ~Plot.label
   ~Plot.limit
   ~Plot.share
   ~Plot.theme

.. rubric:: Integration methods

.. autosummary::
   :toctree: ./
   :nosignatures:

   ~Plot.on

.. rubric:: Output methods

.. autosummary::
   :toctree: ./
   :nosignatures:

   ~Plot.plot
   ~Plot.save
   ~Plot.show

{% endblock %}

.. _plot_config:

Configuration
~~~~~~~~~~~~~

The :class:`Plot` object's default behavior can be configured through its :attr:`Plot.config` attribute. Notice that this is a property of the class, not a method on an instance.

.. include:: ../docstrings/objects.Plot.config.rst
