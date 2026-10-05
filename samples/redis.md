# Redis

Redis is an in-memory data store commonly used for caching, short-lived state, queues, counters, and coordination.

A Redis key can have a TTL, after which Redis automatically removes it. TTLs are useful for data such as sessions or temporary locks that should not live forever.

Pub/sub allows publishers to send messages to subscribers without storing those messages as durable records for later consumers. It is useful for transient notifications.

Redis sets are useful when membership and uniqueness matter. For example, a set can track which worker IDs have reported themselves without adding the same worker twice.
