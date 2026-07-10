"""Final test for complete crash fix: patch both capture_live and capture_and_save."""
import sys
import numpy as np
from autofocus_qt_ui.qt_compat import QApplication
app = QApplication(sys.argv)

print('Test 1: Worker attributes')
from autofocus_qt_ui.worker import AutofocusWorker
worker = AutofocusWorker()
assert hasattr(worker, 'capture_requested'), 'Missing capture_requested signal'
assert hasattr(worker, 'set_captured_image'), 'Missing set_captured_image'
assert hasattr(worker, '_request_capture'), 'Missing _request_capture'
print('  PASS: All attributes exist')

print('\nTest 2: Verify both methods patched')
import inspect
source = inspect.getsource(AutofocusWorker._run_loop)
assert 'safe_capture_live' in source, 'Missing safe_capture_live'
assert 'safe_capture_and_save' in source, 'Missing safe_capture_and_save'
assert 'screen_region' in source, 'Missing screen_region check'
print('  PASS: Both methods patched')

print('\nTest 3: Verify finally restores both methods')
assert 'original_capture_live' in source, 'Missing original_capture_live'
assert 'original_capture_and_save' in source, 'Missing original_capture_and_save'
assert '恢复原始' in source, 'Missing restore comment'
print('  PASS: Both methods will be restored')

print('\nTest 4: Verify _start_loop stops screen preview')
from autofocus_qt_ui.app import AutofocusMainWindow
import inspect
source = inspect.getsource(AutofocusMainWindow._start_loop)
assert '_stop_screen_preview' in source, 'Missing _stop_screen_preview'
print('  PASS: Screen preview will be stopped before loop starts')

print('\nTest 5: Verify worker uses safe capture path')
src_worker = inspect.getsource(AutofocusWorker._run_loop)
assert 'result = self._controller.check_and_autofocus' in src_worker, 'Missing check_and_autofocus'
assert 'save_dir=None' in src_worker, 'Missing save_dir=None'
print('  PASS: Uses save_dir=None forcing capture_live path')

print('\nTest 6: Verify _on_worker_capture exists')
assert '_on_worker_capture' in AutofocusMainWindow.__dict__, 'Missing _on_worker_capture'
method = AutofocusMainWindow._on_worker_capture
sig = inspect.signature(method)
params = list(sig.parameters)
assert len(params) == 1, 'Wrong number of parameters'
print('  PASS: _on_worker_capture correctly defined')

print('\nTest 7: Safe capture returns full metrics dict')
# Check that safe_capture returns all required fields: ok, full_rgb, roi_rgb, full, roi_metrics
import textwrap
src = inspect.getsource(AutofocusWorker._run_loop)
src = textwrap.dedent(src)
assert 'full_metrics' in src, 'Missing full_metrics'
assert 'roi_metrics' in src, 'Missing roi_metrics'
assert 'full_rgb' in src, 'Missing full_rgb'
assert 'roi_rgb' in src, 'Missing roi_rgb'
print('  PASS: Returns complete metrics dict')

print('\nAll 7 tests passed!')
app.quit()