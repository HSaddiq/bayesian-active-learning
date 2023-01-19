def test_version():
    from bal import __version__

    assert isinstance(__version__, str)
