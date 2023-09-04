create table if not exists models(
	instance_uid INTEGER PRIMARY KEY,
	dir TEXT UNIQUE NOT NULL,
	model TEXT NOT NULL
);
