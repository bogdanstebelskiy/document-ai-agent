# SQLite

SQLite is an embedded relational database stored in a file rather than running as a separate database server. It is convenient for small applications, local tools, and tests.

A primary key identifies a row. `UNIQUE` can enforce uniqueness on another column or group of columns. Foreign keys can express relationships between tables.

SQLite supports transactions so several changes can succeed or fail as a unit. A parameterized query should be used instead of constructing SQL by concatenating user input.

`INSERT ... ON CONFLICT DO UPDATE` is useful for an upsert: insert a row when its key is new, or update the existing row when the key already exists.
