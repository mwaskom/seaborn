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

   .. rubric:: Grid methods

   .. autosummary::
      :toctree: ./
      :nosignatures:

      facet
      pair

   .. rubric:: Customization methods

   .. autosummary::
      :toctree: ./
      :nosignatures:

      layout
      label
      limit
      on
      share
      theme

   .. rubric:: Output methods

   .. autosummary::
      :toctree: ./
      :nosignatures:

      plot
      save
      show

   {% endblock %}
