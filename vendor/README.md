# Browser dependency

`plotly-strict-2.33.0.min.js` is the unmodified Plotly.js 2.33.0 strict bundle
from https://cdn.plot.ly/plotly-strict-2.33.0.min.js. It provides the same trace
types and version as the previous CDN dependency while avoiding function
constructors, so the arena can use a Content Security Policy without
`unsafe-eval`. It is served locally to remove runtime CDN script trust.

Plotly's MIT license is in `plotly-LICENSE`.
