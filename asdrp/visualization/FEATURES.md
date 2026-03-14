# Visualization Module Features

Complete feature list for the `asdrp.visualization` module.

## Module Overview

The visualization module provides three main components:
1. **overlay.py** - Real-time pose drawing on video frames
2. **plots.py** - Publication-quality metric plotting
3. **report.py** - Comprehensive report generation

---

## overlay.py Features

### Core Capabilities

✓ **Pose Landmark Visualization**
- Draw 33 MediaPipe pose landmarks
- Customizable colors and sizes
- Visibility-based filtering
- Highlight specific landmarks

✓ **Skeleton Connections**
- Draw skeleton structure
- MediaPipe-compatible connections
- Custom connection sets
- Configurable line thickness

✓ **Gait Event Marking**
- Foot strike visualization
- Toe-off markers
- Mid-stance indicators
- Mid-swing markers
- Color-coded by event type
- Labeled annotations

✓ **Angle Measurements**
- Draw angles between 3 landmarks
- Display angle values
- Custom labels
- Visual arc representation
- Automatic text positioning

✓ **Information Panels**
- Overlay metadata on frames
- Configurable position (4 corners)
- Semi-transparent background
- Multiple data fields
- Automatic sizing

### Technical Features

- Normalized coordinates (0-1) to pixel conversion
- Visibility threshold filtering
- Border drawing for landmarks
- Anti-aliased rendering
- BGR color space support
- Configurable parameters

---

## plots.py Features

### Plot Types

✓ **Time Series Plots**
- Multiple angles simultaneously
- Left/right differentiation (solid/dashed)
- Color-coded by joint
- Gait event markers
- Grid and legends
- Custom time ranges

✓ **Gait Cycle Plots**
- Normalized 0-100% cycle
- Multiple cycles overlay
- Mean ± standard deviation bands
- Individual cycle traces
- Stance/swing phase regions
- Toe-off marker

✓ **Symmetry Analysis**
- Three-panel layout:
  1. Time series comparison
  2. Scatter with line of identity
  3. Difference histogram
- Correlation analysis
- Asymmetry quantification
- Statistical annotations

✓ **Stride Metrics**
- Box plots with statistics
- Individual data points overlay
- Mean and SD display
- Coefficient of variation
- Multiple metrics side-by-side
- Summary text boxes

✓ **Correlation Heatmaps**
- Full correlation matrix
- Annotated values
- Center-normalized colors
- Publication-ready formatting

✓ **Multi-Condition Comparisons**
- Violin plots
- Box plots with swarm overlay
- Statistical comparisons
- Side-by-side conditions

### Visualization Features

- Automatic color scheme selection
- Publication-quality output (150-300 DPI)
- Multiple export formats (PNG, PDF, SVG)
- Seaborn styling
- Responsive legends
- Grid customization
- Title and axis formatting

### Statistical Elements

- Mean lines
- Standard deviation bands
- Confidence intervals
- Individual data points
- Distribution shapes
- Outlier identification

---

## report.py Features

### HTML Report

✓ **Professional Layout**
- Responsive design
- Print-friendly styling
- Gradient header
- Color-coded sections
- Modern card-based metrics
- Shadow effects

✓ **Content Sections**
1. Header with metadata
2. Executive summary
3. Key metrics cards (visual)
4. Embedded visualizations
5. Detailed data tables
6. Interpretation and recommendations
7. Footer with attribution

✓ **Metric Cards**
- Large value display
- Unit labels
- Color gradients
- Grid layout
- Responsive sizing

✓ **Embedded Figures**
- Base64-encoded images
- High-resolution rendering
- Automatic sizing
- Figure captions
- Organized by category

✓ **Data Tables**
- Sortable columns
- Alternating row colors
- Hover effects
- Header styling
- Responsive layout

✓ **Alert Boxes**
- Info alerts (blue)
- Warning alerts (yellow)
- Success alerts (green)
- Custom messages
- Icon support

### Text Summary

✓ **Structured Output**
- ASCII art borders
- Section headers
- Metric summaries with units
- Range information (min-max)
- Recommendations list
- Timestamp and author

✓ **Content Coverage**
- Temporal metrics
- Kinematic metrics
- Symmetry analysis
- Key findings
- Actionable recommendations

### Intelligent Features

✓ **Automatic Interpretation**
- Cadence recommendations (optimal range 170-180 spm)
- Symmetry assessment (good >95%, concerning <90%)
- Contextual advice
- Evidence-based recommendations

✓ **Metric Processing**
- Nested dictionary handling
- Mean ± SD formatting
- Range display (min-max)
- Unit consistency
- Missing data handling

---

## Common Features Across All Modules

### Usability

✓ **Pythonic API**
- Clear function names
- Sensible defaults
- Optional parameters
- Type hints throughout
- Comprehensive docstrings

✓ **Flexibility**
- Customizable colors
- Adjustable sizes
- Optional components
- Format choices
- Path handling

✓ **Error Handling**
- Graceful degradation
- Boundary checking
- Visibility validation
- Missing data handling
- Clear error messages

### Performance

✓ **Efficiency**
- Vectorized operations
- Minimal memory footprint
- Automatic cleanup
- Lazy evaluation
- Batch processing support

✓ **Scalability**
- Handle multiple cycles
- Process long videos
- Large datasets
- Multiple conditions
- Batch reports

### Quality

✓ **Production-Ready**
- Publication-quality output
- High DPI support
- Vector format export
- Color-blind friendly palettes
- Professional styling

✓ **Documentation**
- Complete docstrings
- Type hints
- Usage examples
- API reference
- Quick start guide

---

## Integration Features

### MediaPipe Compatibility

✓ **Direct Support**
- 33-landmark format
- Standard connections
- Visibility scores
- Normalized coordinates
- Named landmark access

### Pandas Integration

✓ **DataFrame Support**
- Column-based access
- Time series data
- Group operations
- Statistical functions
- Export compatibility

### OpenCV Integration

✓ **Video Processing**
- Frame-by-frame overlay
- BGR color space
- Drawing primitives
- Video I/O
- Codec support

### Matplotlib/Seaborn

✓ **Advanced Plotting**
- Figure objects
- Multiple axes
- GridSpec layouts
- Style sheets
- Custom color palettes

---

## Output Formats

### Images
- PNG (recommended for web)
- JPEG (smaller size)
- PDF (vector, publication)
- SVG (vector, web)
- TIFF (high quality)

### Reports
- HTML (interactive viewing)
- Text (quick summary)
- Self-contained (embedded assets)

### Data
- Embedded in figures
- Tables in HTML
- Exportable DataFrames

---

## Customization Options

### Colors
- Landmark colors
- Connection colors
- Side colors (left/right)
- Joint colors (hip/knee/ankle)
- Event colors
- Theme colors

### Sizes
- Figure dimensions
- DPI/resolution
- Landmark radius
- Line thickness
- Font sizes
- Panel sizes

### Styles
- Plot styles (10+ options)
- Color palettes (categorical/sequential)
- Grid styles
- Legend positions
- Layout options

### Content
- Which metrics to display
- Which plots to include
- Report sections
- Annotation detail
- Summary length

---

## Best Practices Supported

### Scientific Rigor
✓ Error bars and uncertainty
✓ Statistical annotations
✓ Reproducible outputs
✓ Documented methods
✓ Version tracking

### Clarity
✓ Clear labels
✓ Appropriate scales
✓ Legends and keys
✓ Color contrast
✓ Readable fonts

### Efficiency
✓ Batch processing
✓ Memory management
✓ File organization
✓ Reusable components
✓ Minimal redundancy

---

## Future-Ready

### Extensibility
- Modular design
- Class inheritance
- Plugin architecture
- Custom renderers
- Format adapters

### Maintainability
- Clean code structure
- Comprehensive tests
- Type safety
- Documentation
- Version control

---

## Summary Statistics

- **3 main modules** (overlay, plots, report)
- **1,760 lines of code**
- **25+ visualization functions**
- **10+ plot types**
- **33 pose landmarks supported**
- **4 gait event types**
- **Multiple output formats**
- **Publication-quality output**

---

## Dependencies

Required:
- OpenCV (cv2)
- NumPy
- Pandas
- Matplotlib
- Seaborn

Optional:
- Jupyter (for notebooks)
- Pillow (for additional formats)
