{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

   .. rubric:: Specification methods

   .. autosummary::
      :toctree: ./
      :nosignatures:

      add
      scale

   .. rubric:: Subplot methods

   .. autosummary::
      :toctree: ./
      :nosignatures:

      facet
      pair
      on

   .. rubric:: Customization methods

   .. autosummary::
      :toctree: ./
      :nosignatures:

      theme
      layout
      label
      limit
      share

   .. rubric:: Output methods

   .. autosummary::
      :toctree: ./
      :nosignatures:

      plot
      save
      show

   {% endblock %}
