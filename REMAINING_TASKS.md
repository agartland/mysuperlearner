# Remaining Documentation Tasks for v0.2.0

**Status:** Core implementation complete (42/42 tests passing)
**Remaining:** Documentation updates and QMD rendering

## ✅ Completed

1. **Phase 1: Core Algorithmic Fixes + sklearn API** - DONE
   - Fixed NNLogLik logit-scale optimization
   - Renamed `ExtendedSuperLearner` → `SuperLearner`
   - Renamed `evaluate_super_learner_cv` → `CVSuperLearner` (class)
   - sklearn-compatible API (learners in constructor)
   - Flexible CV strategy (GroupKFold, TimeSeriesSplit support)

2. **Phase 2: Screening and Composition** - DONE
   - Implemented `VariableSet`, `CorrelationScreener`, `LassoScreener`
   - Works seamlessly with sklearn `Pipeline`

3. **Documentation Started:**
   - ✅ `MIGRATION_GUIDE_v0.2.0.md` created
   - ✅ `mysuperlearner/example_usage.py` updated and tested
   - ✅ All test files updated

## 📋 Remaining Tasks

### 1. Update Python Example Files

#### File: `docs/examples/variable_importance_example.py`

**Changes Needed:**
```python
# OLD imports
from mysuperlearner import ExtendedSuperLearner

# NEW imports
from mysuperlearner import SuperLearner

# OLD usage
sl = ExtendedSuperLearner(method='nnloglik', folds=5)
sl.fit_explicit(X, y, learners)

# NEW usage
sl = SuperLearner(learners=learners, method='nnloglik', cv=5)
sl.fit(X, y)
```

**Search and replace:**
- `ExtendedSuperLearner` → `SuperLearner`
- `folds=` → `cv=`
- `fit_explicit(X, y, learners)` → `fit(X, y)` (and move learners to constructor)

### 2. Update README.md

**Sections to Update:**

1. **Quick Start Example** (lines ~20-50):
```python
# Update example code to use new API
from mysuperlearner import SuperLearner  # was ExtendedSuperLearner
learners = [...]
sl = SuperLearner(learners=learners, method='nnloglik', cv=5)  # was fit_explicit
sl.fit(X_train, y_train)
```

2. **API Documentation** (if present):
   - Update class names
   - Update method signatures
   - Add new screening framework documentation

3. **Installation/Requirements** (verify no changes needed)

4. **Features List** (add):
   - Feature selection via `VariableSet`, `CorrelationScreener`, `LassoScreener`
   - Flexible CV strategy (supports sklearn CV splitters)
   - Enhanced sklearn compatibility

### 3. Update QMD Files (7 files)

All files in `docs/` directory with `.qmd` extension:

1. **`docs/examples/example_analysis.qmd`**
2. **`docs/examples/TESTING_SUMMARY.qmd`**
3. **`docs/examples/variable_importance_guide.qmd`**
4. **`docs/analysis/python_superlearner.qmd`**
5. **`docs/analysis/r_superlearner.qmd`** (may not need Python updates)
6. **`docs/validation/python_cv_superlearner.qmd`**
7. **`docs/validation/r_cv_superlearner.qmd`** (may not need Python updates)

**For each QMD file:**

1. **Find Python code chunks** (look for ` ```{python}` or ` ```python`)
2. **Update imports:**
   ```python
   # OLD
   from mysuperlearner import ExtendedSuperLearner
   from mysuperlearner.evaluation import evaluate_super_learner_cv

   # NEW
   from mysuperlearner import SuperLearner, CVSuperLearner
   ```

3. **Update SuperLearner usage:**
   ```python
   # OLD
   sl = ExtendedSuperLearner(method='nnloglik', folds=5)
   sl.fit_explicit(X, y, learners)

   # NEW
   sl = SuperLearner(learners=learners, method='nnloglik', cv=5)
   sl.fit(X, y)
   ```

4. **Update CV evaluation:**
   ```python
   # OLD
   results = evaluate_super_learner_cv(X, y, base_learners=learners, ...)

   # NEW
   cv_sl = CVSuperLearner(learners=learners, method='nnloglik', cv=5)
   cv_sl.fit(X, y)
   results = cv_sl.get_results()
   ```

5. **Update parameter names:**
   - `folds=` → `cv=`
   - `base_learners=` → `learners=`

### 4. Re-render QMD Files

**After updating all .qmd files:**

```bash
cd /fh/fast/gilbert_p/agartlan/gitrepo/mysuperlearner

# Set R path (if needed for mixed Python/R documents)
export QUARTO_R="/app/software/R/4.4.2-gfbf-2024a/bin/R"
# OR
export QUARTO_R="/app/software/R/4.4.1-gfbf-2023b/bin/R"

# Set PYTHONPATH to ensure mysuperlearner is found
export PYTHONPATH=/fh/fast/gilbert_p/agartlan/gitrepo/mysuperlearner:$PYTHONPATH

# Render all documentation
quarto render docs/

# Or render individual files to test:
quarto render docs/examples/example_analysis.qmd
quarto render docs/validation/python_cv_superlearner.qmd
```

**Verify:**
- Check for any rendering errors
- Verify Python code chunks execute successfully
- Check generated HTML/PDF output

### 5. Update CHANGELOG (Optional but Recommended)

Create or update `CHANGELOG.md`:

```markdown
# Changelog

## [0.2.0] - 2025-01-30

### 🚨 Breaking Changes
- Renamed `ExtendedSuperLearner` → `SuperLearner`
- Renamed `evaluate_super_learner_cv` function → `CVSuperLearner` class
- Changed parameter name `folds` → `cv`
- Moved `learners` from `fit()` to constructor
- Changed default `trim` from 0.025 to 0.001

### ✨ New Features
- Feature selection framework: `VariableSet`, `CorrelationScreener`, `LassoScreener`
- Flexible CV strategy: Support for sklearn CV splitters (GroupKFold, TimeSeriesSplit)
- Enhanced sklearn compatibility (works in pipelines, GridSearchCV)

### 🐛 Bug Fixes
- Fixed NNLogLik optimization to use logit-scale (matches R SuperLearner)
- More numerically stable meta-learner optimization

### 📚 Documentation
- Added `MIGRATION_GUIDE_v0.2.0.md`
- Updated all examples to new API
- Enhanced docstrings

### ⚠️ Deprecations
- `ExtendedSuperLearner` (use `SuperLearner` instead)
- `evaluate_super_learner_cv` function (use `CVSuperLearner` class instead)
- `fit_explicit()` method (use `fit()` instead)
- All deprecated functionality will be removed in v0.3.0

## [0.1.0] - Previous release
...
```

## Automated Search/Replace Patterns

For efficiency, you can use these sed/grep patterns:

```bash
# Find all QMD files with old API
grep -r "ExtendedSuperLearner" docs/*.qmd
grep -r "fit_explicit" docs/*.qmd
grep -r "evaluate_super_learner_cv" docs/*.qmd

# Preview changes (don't apply yet)
grep -n "ExtendedSuperLearner" docs/examples/example_analysis.qmd
grep -n "folds=" docs/examples/example_analysis.qmd

# Apply changes (BE CAREFUL - review first!)
# Example: Replace ExtendedSuperLearner with SuperLearner
sed -i 's/ExtendedSuperLearner/SuperLearner/g' docs/examples/example_analysis.qmd
```

## Verification Checklist

After completing all updates:

- [ ] All Python example files updated and tested
- [ ] README.md updated with new API
- [ ] All 7 QMD files updated with new API
- [ ] All QMD files successfully re-rendered with quarto
- [ ] No import errors in any documentation
- [ ] All code examples execute successfully
- [ ] Generated HTML/PDF documentation looks correct
- [ ] CHANGELOG.md created/updated
- [ ] Git commit with comprehensive message

## Estimated Time

- Python examples: 30 minutes
- README.md: 30 minutes
- QMD files: 2-3 hours (depending on complexity)
- Re-rendering: 30 minutes
- Testing/verification: 1 hour
- **Total: 4-6 hours**

## Git Commit Message Template

```
Update documentation and examples for v0.2.0 API changes

- Updated all Python examples to use new SuperLearner API
- Updated README.md with new class names and method signatures
- Updated all QMD documentation files (7 files)
- Re-rendered Quarto documentation
- Added MIGRATION_GUIDE_v0.2.0.md
- Updated CHANGELOG.md

All examples tested and working with new API.
All 42 tests passing.
```

## Additional Notes

- **Backward compatibility:** Deprecated aliases remain functional with warnings
- **Test coverage:** All 42 tests updated and passing
- **Breaking changes:** Documented in MIGRATION_GUIDE_v0.2.0.md
- **New features:** Screening framework fully functional and tested

---

**Next Steps:** Start with updating the Python example files, then README.md, then systematically work through the QMD files before final rendering.
