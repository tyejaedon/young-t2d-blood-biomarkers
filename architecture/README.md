# Architecture Diagrams

This folder contains PlantUML sources for the project diagrams.

## Files

- `class_diagram.puml` - Conceptual class diagram of the notebook-driven analysis pipeline
- `sequence_diagram.puml` - Execution flow from setup through biomarker modeling
- `system_architecture.puml` - High-level system architecture from data sources to outputs

## Rendering

If PlantUML is available, render any diagram with a command similar to:

```bash
plantuml architecture/class_diagram.puml
plantuml architecture/sequence_diagram.puml
plantuml architecture/system_architecture.puml
```

You can also open the `.puml` files directly in an IDE with PlantUML support.
