# Contributing to AI Commanders

Thank you for your interest in contributing to AI Commanders! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Install dependencies with `uv sync`
4. Create a feature branch from `main`

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ai-commanders.git
cd ai-commanders

# Install dependencies
uv sync

# Run tests to verify setup
uv run pytest tests/ -v
```

## Making Changes

### Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Keep functions focused and concise
- Add docstrings for public functions and classes

### Testing

- Write tests for new features
- Ensure all tests pass before submitting: `uv run pytest tests/ -v`
- Tests are located in the `tests/` directory

### Physics and Combat

The physics and combat systems are based on Terra Invicta mechanics. When modifying these:

- Ensure Newtonian physics remain consistent
- Reference the ship specifications in `data/fleet_ships.json`
- Consider delta-v budgets and acceleration limits

## Submitting Changes

1. Commit your changes with clear, descriptive messages
2. Push to your fork
3. Open a Pull Request against the `main` branch
4. Describe what your changes do and why

## Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Include tests for new functionality
- Update documentation if needed
- Ensure CI tests pass

## Reporting Issues

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS

## Questions?

Open an issue for questions or discussions about the project.
