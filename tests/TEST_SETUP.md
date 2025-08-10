# Test Setup Guide

This document explains how to set up and run the comprehensive test suite for VR Funscript AI Generator.

## Test Data Configuration

### Automatic Test Data Discovery

The test suite uses a centralized configuration system (`test_config.py`) that automatically discovers test video files:

1. **Environment Variable (Recommended)**:
   ```bash
   export FUNGEN_TEST_DATA_DIR="/path/to/your/test/videos"
   ```

2. **Default Locations** (checked in order):
   - `$FUNGEN_TEST_DATA_DIR` (if set)
   - `~/Downloads/test/` (legacy support)  
   - `tests/test_data/` (project default)

### Test Video Categories

Videos are automatically categorized by file size:
- **Short videos** (< 10MB): Used for smoke tests and quick validation
- **Medium videos** (< 100MB): Used for comprehensive testing
- **Long videos** (> 100MB): Used for performance benchmarks

### No Test Videos Available

If no real video files are found, tests will automatically create dummy files for basic functionality testing. However, real videos are recommended for comprehensive validation.

## Running Tests

### Quick Start

```bash
# Run all tests (recommended)
python tests/run_comprehensive_tests.py --mode comprehensive

# Run specific test categories
python tests/run_comprehensive_tests.py --mode smoke         # Fast basic tests
python tests/run_comprehensive_tests.py --mode unit          # Unit tests
python tests/run_comprehensive_tests.py --mode integration   # Integration tests
python tests/run_comprehensive_tests.py --mode e2e           # GUI/E2E tests
python tests/run_comprehensive_tests.py --mode perf          # Performance tests

# Quick mode (skip slow tests)
python tests/run_comprehensive_tests.py --mode comprehensive --quick

# Generate performance report
python tests/run_comprehensive_tests.py --mode comprehensive --report
```

### Test Categories Explained

1. **Smoke Tests**: Fast validation that core functionality works
2. **Unit Tests**: Test individual components in isolation
3. **Integration Tests**: Test component interactions and data flow
4. **E2E Tests**: Test full GUI workflows and user interactions
5. **Performance Tests**: Benchmark processing speeds and regression detection

## Test Coverage

### Functionality Coverage

- ✅ **All tracking modes**: 2-stage, 3-stage, 3-stage-mixed, oscillation-detector
- ✅ **File output validation**: Funscript structure, msgpack data, video files
- ✅ **Interactive timeline**: Point manipulation, heatmaps, scrubbing
- ✅ **GUI components**: Control panels, video display, file management
- ✅ **Stage data carryover**: Stage 2 to Stage 3 data preservation
- ✅ **CLI processing**: Batch and single video processing
- ✅ **Error handling**: Edge cases and failure scenarios

### Test Structure

```
tests/
├── smoke/              # Basic functionality tests
├── unit/               # Component unit tests  
├── integration/        # Component integration tests
├── e2e/                # End-to-end GUI tests
├── performance/        # Performance benchmarks
├── test_config.py      # Centralized test configuration
└── run_comprehensive_tests.py  # Main test runner
```

## Environment Setup

### Required Dependencies

Tests use the same dependencies as the main application. Ensure you have:
- Python 3.8+
- All application dependencies installed
- `FUNGEN_TESTING=1` environment variable (set automatically by test runner)

### Optional Test Setup

```bash
# Set custom test data directory
export FUNGEN_TEST_DATA_DIR="/your/custom/test/videos"

# Enable verbose test output
export PYTEST_VERBOSE=1

# Set custom timeout for slow tests
export PYTEST_TIMEOUT=600
```

## Test Data Recommendations

### Recommended Test Video Files

For comprehensive testing, provide:
- 2-3 short videos (< 30 seconds, < 10MB each)
- 1-2 medium videos (1-5 minutes, < 100MB each)  
- 1 long video (5+ minutes, for performance testing)

### Video Requirements

- **Format**: MP4, MKV, AVI, or MOV
- **Content**: Any video content is fine for testing
- **Naming**: No specific naming requirements
- **Location**: Place in your configured test data directory

### Security Note

- Test videos are never modified or uploaded
- All processing happens locally
- Generated test artifacts are cleaned up automatically
- Performance results are excluded from git commits

## Troubleshooting

### Common Issues

1. **No test videos found**:
   - Set `FUNGEN_TEST_DATA_DIR` to your video directory
   - Tests will create dummy files if no videos available

2. **Tests timeout**:
   - Use `--quick` flag to skip slow tests
   - Increase timeout with `--timeout` parameter

3. **GUI tests fail**:
   - Ensure display/X11 forwarding is available
   - Some E2E tests require actual GUI interaction

4. **Permission errors**:
   - Ensure test data directory is readable
   - Ensure output directory is writable

### Getting Help

- Check test logs for specific error messages
- Run with `--verbose` for detailed output
- Individual test files can be run directly with pytest
- Performance results help identify system-specific issues

## Contributing

### Adding New Tests

1. Choose appropriate test category (smoke/unit/integration/e2e/performance)
2. Use `test_config.py` functions for video paths
3. Include appropriate pytest markers
4. Update this documentation if needed

### Test Best Practices

- Use centralized test configuration
- Clean up temporary files in teardown
- Skip tests gracefully when resources unavailable
- Include both positive and negative test cases
- Document expected behavior clearly