# Testing

A unit test should usually exercise a small piece of behavior without depending on unrelated external systems. Test data should make the reason for a failure easy to understand.

A fixture can provide reusable setup for tests. In pytest, fixtures are declared with `@pytest.fixture` and can be requested by naming them as test function parameters.

Parameterized tests are useful when the same behavior should be checked against several inputs and expected outputs. They avoid copying nearly identical test functions.

A mock is useful when a test needs to control or observe interaction with a dependency. It should not replace every real dependency automatically; excessive mocking can make tests pass while the integration is broken.
