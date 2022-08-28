{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

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
