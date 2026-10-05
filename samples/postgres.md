# PostgreSQL

PostgreSQL is a server-based relational database with strong support for transactions, constraints, indexes, and complex queries.

An index can make lookups faster by providing a data structure the database can use instead of scanning every row. Indexes also consume storage and make writes somewhat more expensive, so they should support actual query patterns.

A transaction groups database operations into an atomic unit. If something fails, a rollback can undo the changes made within that transaction.

Connection pooling lets an application reuse database connections instead of opening a new connection for every request. Too many active connections can overwhelm a database, so pool size should be controlled.
